"""在固定留出集保存完整输出并评价检测精度；不做 NMS，不把小样本代理集称作 COCO 验证。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from common import save_json, sha256
from compare_simulation import read_dump, read_record
from model_geometry import ModelGeometry, SQUARE
from reference import postprocess


def checked_dataset(directory, limit=None):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["content_disjoint"] or not manifest["labelled_proxy_only"]:
        raise ValueError("Expected a content-disjoint labelled proxy set")
    calibration = {row["sha256"] for row in manifest["calibration"]}
    rows = manifest["validation"]
    if not rows or calibration & {row["sha256"] for row in rows}:
        raise ValueError("Empty or overlapping validation data")
    if limit is not None:
        if not 0 < limit <= len(rows):
            raise ValueError("Invalid validation subset size")
        rows = rows[:limit]
    for row in rows:
        if sha256(directory / "validation" / row["input"]) != row["input_sha256"]:
            raise ValueError("Validation input changed")
    return manifest, rows


def original_boxes(detections, transform):
    """预测框还原到原图；整数 resize 导致 x/y 的真实比例可能不同。"""
    result = np.array(detections, dtype=np.float32, copy=True)
    width, height = transform["original"]
    left, top = transform["padding"]
    sx, sy = transform["scale_xy"]
    if min(width, height, sx, sy) <= 0 or not np.isfinite(result).all():
        raise ValueError("Invalid geometry or nonfinite detections")
    result[:, [0, 2]] = np.clip((result[:, [0, 2]] - left) / sx, 0, width)
    result[:, [1, 3]] = np.clip((result[:, [1, 3]] - top) / sy, 0, height)
    return result


def label_boxes(row):
    labels = np.asarray(row["labels_yolo"], dtype=np.float32).reshape(-1, 5)
    width, height = row["transform"]["original"]
    center = labels[:, 1:3] * [width, height]
    half = labels[:, 3:5] * [width / 2, height / 2]
    return labels[:, 0], np.concatenate((center - half, center + half), axis=1).astype(np.float32)


def evaluate(rows, outputs, config_directory, geometry=SQUARE):
    # 复用锁定版 Ultralytics 的匹配和 AP 积分；无需启动 Validator 的数据加载器／训练路径。
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory.resolve())
    import torch
    import ultralytics
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou
    if ultralytics.__version__ != "8.4.0":
        raise ValueError("Evaluation requires the pinned Ultralytics 8.4.0")
    torch.set_num_threads(2)
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    correct, confidence, predicted, targets, counts = [], [], [], [], []
    if len(rows) != len(outputs):
        raise ValueError("Every validation image must have one complete output")
    for row, dense in zip(rows, outputs):
        detections = original_boxes(postprocess(dense, geometry=geometry)[0], row["transform"])
        # 与常规检测验证一致采用 0.001；显示用 0.25，二者不能混同。
        detections = detections[detections[:, 4] >= 0.001]
        classes, boxes = label_boxes(row)
        iou = box_iou(torch.from_numpy(boxes), torch.from_numpy(detections[:, :4]))
        matches = BaseValidator.match_predictions(matcher, torch.from_numpy(detections[:, 5]),
                                                 torch.from_numpy(classes), iou)
        correct.append(matches.numpy())
        confidence.append(detections[:, 4])
        predicted.append(detections[:, 5])
        targets.append(classes)
        counts.append(dict(case=row["case"], labelled=len(classes),
                           predicted_at_025=int(np.count_nonzero(detections[:, 4] >= 0.25))))
    if not any(len(item) for item in targets):
        raise ValueError("AP is undefined for a set without ground truth")
    metrics = ap_per_class(np.concatenate(correct), np.concatenate(confidence), np.concatenate(predicted),
                           np.concatenate(targets), plot=False)
    ap, classes = metrics[5], metrics[6]
    return dict(images=len(rows), labels=sum(len(item) for item in targets), classes=len(classes),
                map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()),
                map_by_iou=[dict(iou=round(.5 + .05 * index, 2), map=float(value))
                            for index, value in enumerate(ap.mean(axis=0))],
                per_class=[dict(class_id=int(c), ap50=float(a[0]), ap50_95=float(a.mean()))
                           for c, a in zip(classes, ap)], cases=counts, nms=False, max_det=300,
                confidence_filter=0.001, metrics_library="Ultralytics 8.4.0 ap_per_class / match_predictions",
                official_coco_validation=False, task_scene_accuracy_qualified=False)


def collect(dataset, output, onnx=None, model=None, stage="ZG", limit=None, resume=False):
    manifest, rows = checked_dataset(dataset, limit)
    geometry = ModelGeometry(*manifest["target"])
    if (onnx is None) == (model is None):
        raise ValueError("Choose exactly one ONNX reference or Icraft candidate")
    graph = onnx if onnx else model / ("yolo26l_" + stage + ".json")
    identity = dict(dataset_manifest_sha256=sha256(dataset / "manifest.json"), graph_sha256=sha256(graph),
                    raw_sha256=None if onnx else sha256(graph.with_suffix(".raw")),
                    mode="onnx" if onnx else stage, cases=[row["case"] for row in rows],
                    full_held_out_set=len(rows) == len(manifest["validation"]), board_access=False)
    if resume:
        saved = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        if identity != saved:
            raise ValueError("Cannot resume after changing inputs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "identity.json", identity)
    session = None
    if onnx:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("Expected the single-output raw one-to-one reference graph")
    complete, outputs = [], []
    for row in rows:
        case = row["case"]
        name = "case-{}-dense.npy".format(case)
        path, receipt = output / name, output / (name + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved["input_sha256"] != row["input_sha256"] or saved["output_sha256"] != sha256(path):
                raise ValueError("Saved output identity changed")
            dense = np.load(str(path), allow_pickle=False)
        else:
            if path.exists():
                raise ValueError("Unreviewed output exists; preserve it before recovery")
            if session:
                image = np.load(str(dataset / "validation" / row["input"]), allow_pickle=False)
                dense = session.run(None, {session.get_inputs()[0].name: image})[0]
            else:
                leaf = output / ("simulate-{}".format(case))
                if not leaf.exists():
                    subprocess.run([sys.executable, str(Path(__file__).with_name("simulate_model.py")),
                                    "--model", str(model), "--fixtures", str(dataset / "validation"),
                                    "--output", str(leaf), "--stage", stage, "--case", str(case)], check=True)
                native = read_record(leaf)
                if native["graph_sha256"] != identity["graph_sha256"] or native["input_sha256"] != row["input_sha256"]:
                    raise ValueError("Native output is paired with another graph or input")
                dense, _ = read_dump(leaf)
            if dense.shape != geometry.dense or dense.dtype != np.float32 or not np.isfinite(dense).all():
                raise ValueError("Invalid complete output; do not publish detections")
            np.save(str(path), dense)
            save_json(receipt, dict(input_sha256=row["input_sha256"], output_sha256=sha256(path)))
        complete.append(case)
        outputs.append(dense)
        save_json(output / "progress.json", dict(completed_cases=complete, expected=len(rows)))
        print("Saved complete output", case, "({}/{})".format(len(complete), len(rows)), flush=True)
    report = evaluate(rows, outputs, output / "ultralytics-config", geometry)
    report.update(identity=identity)
    save_json(output / "metrics.json", report)
    print("Proxy mAP50-95:", report["map50_95"], "images:", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=Path)
    group.add_argument("--model", type=Path)
    parser.add_argument("--stage", choices=("optimized", "quantized", "ZG"), default="ZG")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    collect(args.dataset.resolve(), args.output.resolve(), args.onnx.resolve() if args.onnx else None,
            args.model.resolve() if args.model else None, args.stage, args.limit, args.resume)


if __name__ == "__main__":
    main()
