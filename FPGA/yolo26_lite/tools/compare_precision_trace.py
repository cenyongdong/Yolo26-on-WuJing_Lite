"""比较 SFB 中间特征；结果含前序累积误差，不冒充单层独立量化误差。"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from common import save_json, sha256
from compare_simulation import read_record


def logical_nhwc(value):
    dtype = value["dtype"]
    shape, layout = dtype["shape"], dtype["layout"]
    if layout == "@layout(NHWC)" and len(shape) == 4:
        return shape
    match = re.fullmatch(r"@layout\(NCHWc(\d+)\)", layout)
    if match and len(shape) == 5 and shape[-1] == int(match.group(1)):
        return [shape[0], shape[2], shape[3], shape[1] * shape[4]]
    raise ValueError("Unsupported trace layout; do not guess a reshape: " + layout)


def tensor_metrics(reference, actual):
    if reference.size != actual.size or reference.size == 0:
        raise ValueError("Trace element counts differ or are empty")
    square_error = square_reference = square_actual = dot = maximum = 0.0
    # float64 分块累计，避免为大特征图同时创建多个完整的双精度副本。
    for start in range(0, reference.size, 262144):
        a = reference[start:start + 262144].astype(np.float64)
        b = actual[start:start + 262144].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite intermediate feature")
        delta = a - b
        square_error += float(np.dot(delta, delta))
        square_reference += float(np.dot(a, a))
        square_actual += float(np.dot(b, b))
        dot += float(np.dot(a, b))
        maximum = max(maximum, float(np.abs(delta).max()))
    denominator = np.sqrt(square_reference * square_actual)
    return dict(max_abs=maximum, rms=float(np.sqrt(square_error / reference.size)),
                relative_l2=float(np.sqrt(square_error / square_reference)) if square_reference else None,
                cosine=float(dot / denominator) if denominator else None)


def compare_traces(reference_directory, candidate_directory):
    reference = read_record(reference_directory)
    candidate = read_record(candidate_directory)
    for key in ("case", "input_sha256", "encoded_input_sha256", "encoded_input_bytes"):
        if reference[key] != candidate[key]:
            raise ValueError("Trace inputs differ: " + key)
    values = {label: {value["v_id"]: value for value in record.get("traced_values", [])}
              for label, record in (("reference", reference), ("candidate", candidate))}
    if not values["reference"] or values["reference"].keys() != values["candidate"].keys():
        raise ValueError("Both simulations must explicitly trace the same value IDs")
    rows = []
    for identifier, value in values["reference"].items():
        shape = logical_nhwc(value)
        if shape != logical_nhwc(values["candidate"][identifier]):
            raise ValueError("Logical intermediate shapes differ")
        arrays, paths = {}, {}
        for label, directory in (("reference", reference_directory), ("candidate", candidate_directory)):
            matches = list(directory.glob("native/*/ftmpSFB/{}.ftmp".format(identifier)))
            if len(matches) != 1 or matches[0].stat().st_size != int(np.prod(shape)) * 4:
                raise ValueError("Missing or incomplete SFB intermediate output")
            paths[label] = matches[0]
            arrays[label] = np.memmap(str(matches[0]), mode="r", dtype="<f4")
        # SFB 已由工具转为软件顺序；再次按 NCHWc 做转置反而打乱数据。
        metrics = tensor_metrics(arrays["reference"], arrays["candidate"])
        rows.append(dict(value_id=identifier, logical_shape=shape, metrics=metrics,
                         reference_layout=value["dtype"]["layout"],
                         candidate_layout=values["candidate"][identifier]["dtype"]["layout"],
                         reference_sha256=sha256(paths["reference"]), candidate_sha256=sha256(paths["candidate"])))
    return dict(schema=1, case=reference["case"], input_sha256=reference["input_sha256"],
                reference_graph_sha256=reference["graph_sha256"], candidate_graph_sha256=candidate["graph_sha256"],
                features=rows, includes_upstream_error=True, isolated_operator_error=False, board_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_traces(args.reference, args.candidate)
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
