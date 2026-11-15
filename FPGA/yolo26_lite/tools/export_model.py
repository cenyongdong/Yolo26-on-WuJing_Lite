"""只读原始权重，导出完整参考图及 one-to-one 原始头部署图并实际运行 ORT。"""
from pathlib import Path
import argparse
import collections
import json
import os
import sys
import time

from common import save_json, sha256
from model_geometry import ModelGeometry


def audit_graph(path):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shapes[value.name] = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    counts = collections.Counter(node.op_type for node in model.graph.node)
    if counts["NonMaxSuppression"]:
        raise ValueError("NMS is forbidden in the one-to-one graph")
    return dict(sha256=sha256(path), bytes=Path(path).stat().st_size,
                operators=dict(sorted(counts.items())),
                outputs={value.name: shapes[value.name] for value in model.graph.output},
                attention=[dict(name=node.name, op=node.op_type,
                                inputs=[shapes.get(name) for name in node.input])
                           for node in model.graph.node if node.op_type in ("MatMul", "Softmax")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, choices=(11, 13), default=11)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    geometry = ModelGeometry(*args.size)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics-config")
    (output / "ultralytics-config").mkdir()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    from reference import check_selected_rows

    if ultralytics.__version__ != "8.4.0":
        raise RuntimeError("Exporter must be pinned to ultralytics 8.4.0")
    torch.set_num_threads(2)
    torch.manual_seed(26)
    started = time.monotonic()
    model = YOLO(str(args.weights.resolve())).model.cpu().float().eval()
    head = model.model[-1]
    if not (head.end2end and head.reg_max == 1 and head.nc == 80 and head.max_det == 300):
        raise ValueError("Unexpected model head identity")
    model.fuse(verbose=False)
    head.export = False
    head.format = "onnx"
    head.dynamic = False

    class Complete(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return self.network(image)[0]

    class Dense(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            # 使用实际 one-to-one 头，不重新实现模型主干或检测卷积。
            raw = self.network(image)[1]["one2one"]
            return torch.cat((raw["boxes"], raw["scores"]), dim=1)

    inputs = [np.full(geometry.nchw, 114 / 255, np.float32),
              np.random.RandomState(26).uniform(0, 1, geometry.nchw).astype(np.float32)]
    asset = ASSETS / "bus.jpg"
    if asset.is_file():
        image = cv2.cvtColor(cv2.imread(str(asset)), cv2.COLOR_BGR2RGB)
        gain = min(geometry.width / image.shape[1], geometry.height / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * gain), round(image.shape[0] * gain)))
        padded = np.full(geometry.nhwc[1:], 114, np.uint8)
        left = (geometry.width - resized.shape[1]) // 2
        top = (geometry.height - resized.shape[0]) // 2
        padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        inputs.append((padded.transpose(2, 0, 1)[None].astype(np.float32) / 255))
    sample = torch.from_numpy(inputs[0])
    complete, dense = Complete(model).eval(), Dense(model).eval()
    with torch.inference_mode():
        full = complete(sample)
        raw = dense(sample)
        if tuple(full.shape) != (1, 300, 6) or tuple(raw.shape) != geometry.dense:
            raise ValueError("Unexpected output shapes")
        for name, wrapper, output_name in (("complete", complete, "detections"),
                                            ("dense", dense, "one2one_raw")):
            path = output / ("yolo26l_" + name + ".onnx")
            torch.onnx.export(wrapper, sample, str(path), export_params=True, opset_version=args.opset,
                              do_constant_folding=True, input_names=["images"], output_names=[output_name])
            save_json(output / (name + ".graph.json"), audit_graph(path))
            print("exported", name, flush=True)

    if any(audit_graph(output / "yolo26l_dense.onnx")["operators"].get(op, 0)
           for op in ("TopK", "GatherElements", "NonMaxSuppression")):
        raise AssertionError("Deployment cut still contains selection operators")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(output / ("yolo26l_" + name + ".onnx")), options,
                                         providers=["CPUExecutionProvider"])
                for name in ("complete", "dense")}
    comparisons = []
    for index, input_value in enumerate(inputs):
        with torch.inference_mode():
            expected_full = complete(torch.from_numpy(input_value)).numpy()
            expected_dense = dense(torch.from_numpy(input_value)).numpy()
        actual_full = sessions["complete"].run(None, {"images": input_value})[0]
        actual_dense = sessions["dense"].run(None, {"images": input_value})[0]
        # 先保存完整原值，失败也可以复现。
        for name, value in (("input", input_value), ("torch-complete", expected_full),
                            ("torch-dense", expected_dense), ("ort-complete", actual_full),
                            ("ort-dense", actual_dense)):
            np.save(str(output / ("case-{}-{}.npy".format(index, name))), value)
        np.testing.assert_allclose(actual_dense, expected_dense, rtol=1e-3, atol=0.0002)
        check_selected_rows(expected_dense, expected_full, geometry=geometry)
        check_selected_rows(actual_dense, actual_full, geometry=geometry)
        np.testing.assert_allclose(actual_full[0, :, 4], expected_full[0, :, 4], rtol=1e-3, atol=0.0002)
        # 高分真实检测还必须保持类别和框；低分并列只按候选及分值契约校验。
        high_expected = expected_full[0][expected_full[0, :, 4] >= 0.25]
        high_actual = actual_full[0][actual_full[0, :, 4] >= 0.25]
        np.testing.assert_allclose(high_actual, high_expected, rtol=1e-3, atol=0.01)
        comparisons.append(dict(case=index, dense_max_abs=float(np.max(np.abs(actual_dense - expected_dense))),
                                score_max_abs=float(np.max(np.abs(actual_full[:, :, 4] - expected_full[:, :, 4]))),
                                detections_at_025=int(np.sum(actual_full[:, :, 4] >= 0.25))))
        print("validated", comparisons[-1], flush=True)
    import importlib.metadata
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "numpy", "opencv-python")}
    save_json(output / "export-result.json", dict(status="onnx_and_reference_passed", versions=versions,
              python=sys.version, weights=str(args.weights.resolve()), weights_sha256=sha256(args.weights),
              shapes=dict(input=geometry.nchw, dense=geometry.dense, complete=[1, 300, 6]),
              branch="one2one", nms=False, opset=args.opset, preprocessing="RGB float32 /255 NCHW",
              dense_contract="LTRB distances in stride units, then 80 class logits; P3,P4,P5 row-major",
              comparisons=comparisons, elapsed_seconds=time.monotonic() - started,
              board_tested=False, quantized_accuracy_validated=False))


if __name__ == "__main__":
    main()
