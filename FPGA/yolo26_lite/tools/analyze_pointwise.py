"""沿相同输入／输出边界汇总 1×1 变换区域，纳入 Copy、Transpose 和其它新增节点。"""
import argparse
from pathlib import Path

from analyze_instructions import analyze_text
from analyze_layout import kind
from analyze_npu import read_json, require, tensor_bytes
from common import save_json, sha256


def region(graph, input_value, output_value):
    return region_between(graph, [input_value], [output_value])


def region_between(graph, input_values, output_values):
    """按共同的多输入／多输出边界比较布局传播；共享节点只计一次。

    允许跨分支保留布局后，原单层的输入值可能已被消除。此时应扩大到两图
    都存在的边界，包含生产者、真实拼接和恢复分支，不能漏算移动到上游的转换。
    """
    inputs, outputs = set(input_values), set(output_values)
    require(inputs and outputs and len(inputs) == len(input_values) and len(outputs) == len(output_values),
            "Region boundaries must be nonempty and distinct")
    require(not inputs & outputs, "Region input and output boundaries overlap")
    producers = {v["v_id"]: op for op in graph["ops"] for v in op["outputs"]}
    require(len(producers) == sum(len(op["outputs"]) for op in graph["ops"]),
            "Region graph has multiple producers for a value")
    found, active, reached = {}, set(), set()

    def visit(identity):
        if identity in inputs:
            reached.add(identity)
            return
        require(identity not in active and identity in producers, "Region contains a cycle or missing producer")
        op = producers[identity]
        require(kind(op) not in ("Input", "Output"), "Region escaped the selected input boundary")
        if op["op_id"] in found:
            return
        active.add(identity)
        for value in op["inputs"]:
            visit(value["v_id"])
        active.remove(identity)
        found[op["op_id"]] = op

    for identity in output_values:
        visit(identity)
    require(reached == inputs, "Region never reached every input boundary")
    return [op for op in graph["ops"] if op["op_id"] in found]


def pointwise_coverage(op, requests):
    """核对普通单输入点卷积的指令覆盖，避免布局被接受却只处理部分张量。

    ETM 槽按输入、参数、输出排列；覆盖字节相等是必要条件，不能代替像素顺序和数值验证。
    """
    require(kind(op) == "Conv2d" and len(op["inputs"]) == len(op["outputs"]) == 1,
            "Coverage requires a single-input/output convolution")
    output_slot = len(op["inputs"]) + len(op["params"])
    rows = {(row["direction"], row["buffer_slot"]): row for row in requests["memory"]}
    result = {}
    for role, direction, slot, value in (("input", "RETM", 0, op["inputs"][0]),
                                         ("output", "WETM", output_slot, op["outputs"][0])):
        require((direction, slot) in rows, "Missing pointwise memory operand")
        actual = rows[(direction, slot)]["unique_address_bytes"]
        expected = tensor_bytes(value)
        require(actual == expected, "Pointwise {} coverage is {} of {} bytes".format(role, actual, expected))
        result[role] = dict(value_id=value["v_id"], expected_bytes=expected, unique_address_bytes=actual)
    return result


def analyze_region(graph, baseline_op, logs, timings=None):
    start, end = baseline_op["inputs"][0]["v_id"], baseline_op["outputs"][0]["v_id"]
    operations = region(graph, start, end)
    rows = []
    for op in operations:
        row = dict(operation=op["op_id"], kind=kind(op), layer_count=op["layer_count"])
        if op["layer_count"]:
            path = logs / ("hardOp" + str(op["op_id"]) + "_zg_insts.log")
            row.update(log_sha256=sha256(path), requests=analyze_text(path.read_text(encoding="utf-8")))
            if kind(op) == "Conv2d":
                row["core_memory_coverage"] = pointwise_coverage(op, row["requests"])
        if timings is not None:
            require(op["op_id"] in timings, "Region timing is missing")
            row["mean_ms"] = timings[op["op_id"]]
        rows.append(row)
    return dict(original_operation=baseline_op["op_id"], input_value=start, output_value=end, operations=rows,
                requested_read_bytes=sum(r.get("requests", {}).get("requested_read_bytes", 0) for r in rows),
                requested_write_bytes=sum(r.get("requests", {}).get("requested_write_bytes", 0) for r in rows),
                mean_ms=sum(r["mean_ms"] for r in rows) if timings is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--operation", type=int, action="append", required=True)
    parser.add_argument("--timing-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous region analysis")
    baseline_path, candidate_path = (p / "yolo26l_ZG.json" for p in (args.baseline, args.candidate))
    baseline, candidate = read_json(baseline_path), read_json(candidate_path)
    ops = {op["op_id"]: op for op in baseline["ops"]}
    timing = None
    if args.timing_report:
        report = read_json(args.timing_report)
        matches = [r for r in report["runs"] if r["identities"]["model.json"] == sha256(candidate_path)]
        require(len(matches) == 1 and matches[0]["timer_semantics"] == "durations", "Unique verified timing required")
        timing = {row["operation"]: row["milliseconds"]["mean"] for row in matches[0]["ranked_operations"]}
    rows = []
    for identity in args.operation:
        require(identity in ops and kind(ops[identity]) == "Conv2d", "Expected an original convolution")
        rows.append(analyze_region(candidate, ops[identity], args.candidate / "logs/generate/yolo26l", timing))
    save_json(args.output, dict(schema=1, baseline_graph_sha256=sha256(baseline_path),
              candidate_graph_sha256=sha256(candidate_path), regions=rows, actual_ddr_bytes_measured=False))
    for row in rows:
        print(row["original_operation"], "total requested MiB:",
              (row["requested_read_bytes"] + row["requested_write_bytes"]) / 2**20,
              "region ms:", row["mean_ms"])


if __name__ == "__main__":
    main()
