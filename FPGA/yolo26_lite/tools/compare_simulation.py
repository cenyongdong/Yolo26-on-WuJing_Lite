"""比较 FP16 Host／生成代码模拟与浮点参考；仅作 fixture 回归，不代替 mAP。"""
from pathlib import Path
import argparse
import json

import numpy as np
from common import save_json, sha256
from model_geometry import ModelGeometry, SQUARE
from reference import postprocess
from simulate_model import encode_input


def read_record(directory):
    record = json.loads((directory / "simulation.json").read_text(encoding="utf-8"))
    if record["run"]["native_exit"] != 0:
        raise ValueError("Simulation did not finish successfully")
    identifiers = record["output_value_ids"]
    if len(identifiers) != 1 or type(identifiers[0]) is not int or identifiers[0] < 0:
        raise ValueError("Expected one output tensor identifier")
    return record


def read_dump(directory):
    record = read_record(directory)
    geometry = ModelGeometry(*record.get("input_size", (640, 640)))
    elements = geometry.dense[1] * geometry.anchors
    identifier = record["output_value_ids"][0]
    paths = list(directory.glob("native/*/ftmpSFB/{}.ftmp".format(identifier)))
    if len(paths) != 1 or paths[0].stat().st_size != elements * 4:
        raise ValueError("Expected exactly one complete output dump")
    raw = np.fromfile(str(paths[0]), dtype="<f4")
    if raw.size != elements or not np.isfinite(raw).all():
        raise ValueError("Invalid simulation output")
    return raw.reshape(geometry.dense), paths[0]


def verify_input_pair(input_path, host_directory, generated_directory):
    """数值比较前核对同一原图及实际输入字节；旧 FP32 记录可由原图重建编码。"""
    host = read_record(host_directory)
    generated = read_record(generated_directory)
    case = host["case"]
    if type(case) is not int or case < 0 or type(generated["case"]) is not int or generated["case"] != case:
        raise ValueError("Host and generated simulations use different fixture cases")
    if host["stage"] not in ("optimized", "quantized") or generated["stage"] != "ZG":
        raise ValueError("Expected a Host reference and a generated ZG simulation")
    input_identity = sha256(input_path)
    if host["input_sha256"] != input_identity or generated["input_sha256"] != input_identity:
        raise ValueError("Simulation fixture identity differs from the reference input")
    image = np.load(str(input_path), allow_pickle=False)
    if image.ndim != 4:
        raise ValueError("Expected an NCHW image fixture")
    geometry = ModelGeometry(image.shape[3], image.shape[2])
    identities = {}
    for name, directory, record in (("host", host_directory, host),
                                     ("generated", generated_directory, generated)):
        element_type = record.get("input_element_type", "@fp(32)")
        if ModelGeometry(*record.get("input_size", (640, 640))) != geometry:
            raise ValueError("Simulation geometry differs from the reference input")
        graph = dict(ops=[dict(_type_key="icraft::xir::Input", outputs=[dict(dtype=dict(
            shape=geometry.nhwc, layout="@layout(NHWC)", element_dtype=element_type))])])
        expected, _ = encode_input(image, graph)
        encoded_path = directory / "input.ftmp"
        encoded_identity = sha256(encoded_path)
        if (encoded_path.stat().st_size != expected.nbytes or
                encoded_path.read_bytes() != expected.tobytes(order="C") or
                record.get("encoded_input_bytes", expected.nbytes) != expected.nbytes or
                record.get("encoded_input_sha256", encoded_identity) != encoded_identity):
            raise ValueError("{} encoded input does not match its original fixture".format(name))
        identities[name] = dict(graph_sha256=record["graph_sha256"], input_element_type=element_type,
                                encoded_input_sha256=encoded_identity, encoded_input_bytes=expected.nbytes)
    return dict(case=case, reference_input_sha256=input_identity,
                input_size=[geometry.width, geometry.height], **identities)


def compare(reference, actual, geometry=SQUARE):
    for label, tensor in (("reference", reference), ("actual", actual)):
        if tensor.shape != geometry.dense or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError("Invalid FP32 dense tensor: " + label)
    difference = np.abs(reference - actual)
    worst = tuple(int(value) for value in np.unravel_index(np.argmax(difference), difference.shape))
    expected_boxes = postprocess(reference, geometry=geometry)[0]
    actual_boxes = postprocess(actual, geometry=geometry)[0]
    expected_boxes = expected_boxes[expected_boxes[:, 4] >= 0.25]
    actual_boxes = actual_boxes[actual_boxes[:, 4] >= 0.25]
    matches = []
    unmatched = set(range(len(actual_boxes)))
    for expected in expected_boxes:
        candidates = []
        for index in unmatched:
            found = actual_boxes[index]
            if found[5] != expected[5]:
                continue
            extent = np.maximum(np.minimum(found[2:4], expected[2:4]) -
                                np.maximum(found[:2], expected[:2]), 0)
            intersection = float(np.prod(extent))
            union = float(np.prod(found[2:4] - found[:2]) + np.prod(expected[2:4] - expected[:2])) - intersection
            candidates.append((intersection / union if union > 0 else 0, index))
        if not candidates:
            matches.append(dict(class_id=int(expected[5]), matched=False))
            continue
        overlap, index = max(candidates)
        unmatched.remove(index)
        found = actual_boxes[index]
        matches.append(dict(class_id=int(expected[5]), matched=True, iou=overlap,
                            max_coordinate_error=float(np.max(np.abs(found[:4] - expected[:4]))),
                            score_error=float(abs(found[4] - expected[4]))))
    passed = (len(expected_boxes) == len(actual_boxes) and not unmatched and
              all(row.get("matched") and row["iou"] >= 0.98 and row["score_error"] <= 0.02 for row in matches))
    return dict(raw_max_abs=float(difference.max()), raw_rms=float(np.sqrt(np.mean(difference ** 2))),
                worst_index=worst, worst_reference=float(reference[worst]), worst_actual=float(actual[worst]),
                expected_high_detections=expected_boxes.tolist(), actual_high_detections=actual_boxes.tolist(),
                high_detection_matches=matches, fixture_passed=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-input", type=Path,
                        help="Defaults to case-N-input.npy beside the reference output")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.reference_input
    if input_path is None:
        case = read_record(args.host)["case"]
        input_path = args.reference.parent / "case-{}-input.npy".format(case)
    pairing = verify_input_pair(input_path, args.host, args.generated)
    geometry = ModelGeometry(*pairing["input_size"])
    reference = np.load(str(args.reference), allow_pickle=False)
    host, host_path = read_dump(args.host)
    generated, generated_path = read_dump(args.generated)
    result = dict(input_pairing=pairing, reference_sha256=sha256(args.reference), host_sha256=sha256(host_path),
                  generated_sha256=sha256(generated_path), host=compare(reference, host, geometry),
                  generated=compare(reference, generated, geometry), generated_vs_host=compare(host, generated, geometry),
                  board_tested=False, map_validated=False,
                  acceptance="fixture only: same high-score classes/count, matched IoU>=0.98 and score delta<=0.02")
    # 不通过也先保存原始差异和最差坐标，不以摘要覆盖失败。
    save_json(args.output, result)
    if not all(result[key]["fixture_passed"] for key in ("host", "generated", "generated_vs_host")):
        raise AssertionError("Simulation fixture comparison failed; see " + str(args.output))
    print("PASS fixture comparisons; mAP and board performance remain unverified", flush=True)


if __name__ == "__main__":
    main()
