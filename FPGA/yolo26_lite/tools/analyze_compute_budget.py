"""按逻辑图核算 Conv/Matmul 的乘加量；峰值预算不代表板端性能或完整模型延迟。"""
import argparse
from collections import defaultdict
from itertools import zip_longest
import math
from pathlib import Path

from analyze_layout import kind
from common import save_json, sha256
from analyze_npu import read_json, require


def shape(value):
    result = value["dtype"]["shape"]
    require(result and all(type(n) is int and n > 0 for n in result), "Expected positive static geometry")
    return result


def storage(value):
    element = value["dtype"]["element_dtype"]
    return element["storage_dtype"] if isinstance(element, dict) else element


def operation_count(operation):
    """一次乘加按两次运算；卷积权重 I 维已经是每组输入通道，不能再次除以 groups。"""
    name = kind(operation)
    if name not in ("Conv2d", "Matmul"):
        return None
    left, right = operation["inputs"][:2]
    output = operation["outputs"][0]
    a, b, z = shape(left), shape(right), shape(output)
    precision = storage(left)
    require(storage(right) == precision and precision in ("@fp(16)", "@sint(8)", "@fp(32)"),
            "Different operand encodings need separate arithmetic accounting")
    if name == "Conv2d":
        require(left["dtype"]["layout"] == output["dtype"]["layout"] == "@layout(NHWC)" and
                right["dtype"]["layout"] == "@layout(HWIO)" and len(a) == len(b) == len(z) == 4,
                "Use the logical NHWC/HWIO graph, not padded hardware tensors")
        groups = operation["groups"]
        require(type(groups) is int and groups > 0 and a[0] == z[0] and
                b[2] * groups == a[3] and b[3] == z[3] and z[3] % groups == 0,
                "Convolution channel/group contract differs")
        return precision, 2 * z[0] * z[1] * z[2] * math.prod(b)
    require(len(a) >= 2 and len(b) >= 2 and a[-1] == b[-2], "Invalid matrix contraction")
    leading = []
    for x, y in zip_longest(reversed(a[:-2]), reversed(b[:-2]), fillvalue=1):
        require(x == y or x == 1 or y == 1, "Incompatible matrix broadcasting")
        leading.append(max(x, y))
    expected = list(reversed(leading)) + [a[-2], b[-1]]
    require(z == expected, "Matrix output geometry differs from its contraction")
    return precision, 2 * math.prod(z[:-2]) * a[-2] * a[-1] * b[-1]


def analyze(graph, int8_tops=8.0, fp16_tflops=4.0):
    require(all(math.isfinite(v) and v > 0 for v in (int8_tops, fp16_tflops)), "Invalid peak assumptions")
    rows, totals, ignored = [], defaultdict(int), defaultdict(int)
    for op in graph["ops"]:
        counted = operation_count(op)
        if counted is None:
            ignored[kind(op)] += 1
            continue
        precision, operations = counted
        totals[precision] += operations
        rows.append(dict(operation=op["op_id"], kind=kind(op), precision=precision, operations=operations))
    require("@fp(32)" not in totals, "FP32 peak was not specified")
    milliseconds = totals.get("@sint(8)", 0) / (int8_tops * 1e9) + totals.get("@fp(16)", 0) / (fp16_tflops * 1e9)
    return dict(schema=1, operations_per_mac=2, logical_operations=sum(totals.values()),
                operations_by_precision=dict(totals), operations=rows, excluded_operator_counts=dict(ignored),
                assumed_int8_tops=int8_tops, assumed_fp16_tflops=fp16_tflops,
                ideal_conv_matmul_ms=milliseconds, ideal_conv_matmul_fps=1000 / milliseconds if milliseconds else None,
                includes_padding_or_recomputation=False, includes_other_ops_or_transfer=False,
                peak_is_measured=False, board_tested=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing arithmetic evidence")
    report = analyze(read_json(args.graph))
    report["logical_graph_sha256"] = sha256(args.graph)
    save_json(args.output, report)
    print("Logical Conv/Matmul GOP:", report["logical_operations"] / 1e9)
    print("Optimistic arithmetic ms (excludes all transfers and other operators):", report["ideal_conv_matmul_ms"])


if __name__ == "__main__":
    main()
