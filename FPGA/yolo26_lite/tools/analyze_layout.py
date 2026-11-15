"""离线连接原生层计时、卷积循环与 Split/Copy/Concat 存储关系；不修改编译图。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from analyze_npu import digest, read_json, require, tensor_bytes


def kind(op):
    name = op.get("name") or op["_type_key"]
    name = name.split("::")[-1]
    return name[:-4] if name.endswith("Node") else name


def chunk(value):
    match = re.fullmatch(r"@chunk\((\d+)@(\d+)\)", value.get("mtype", ""))
    require(match is not None, "Expected compiler chunk metadata")
    return tuple(int(item) for item in match.groups())


def parse_loops(text):
    """保留编译日志原始循环边界；不把 MPE_t 直接解释成 DDR 重读次数。"""
    result, current = {}, None
    header = re.compile(r"PerformLoopTransform, op_name = (\S+), op_ofm_vid = (\d+), nested_loop")
    loop = re.compile(r"\s*for (.+?), in (.+?)\s*:\s*#\s*(.+)")
    for line in text.splitlines():
        match = header.search(line)
        if match:
            key = (match.group(1).split("::")[-1], int(match.group(2)))
            require(key not in result, "Ambiguous repeated loop transform: " + str(key))
            current = result[key] = []
            continue
        match = loop.fullmatch(line)
        if current is not None and match:
            names = [v.strip() for v in match.group(1).split(",")]
            sizes = [int(v.strip()) for v in match.group(2).split(",")]
            roles = [v.strip() for v in match.group(3).split(",")]
            require(len(names) == len(sizes) == len(roles) and all(v > 0 for v in sizes),
                    "Malformed compiler loop")
            current.append(dict(zip(names, sizes)))
        elif current is not None and line.strip():
            current = None
    return result


def convolution_rows(generated, adapted, quantized, loops, timing):
    adapted_by_id = {op["op_id"]: op for op in adapted["ops"]}
    quantized_by_output = {op["outputs"][0]["v_id"]: op for op in quantized["ops"]
                           if kind(op) == "Conv2d"}
    rows = []
    for op in generated["ops"]:
        if kind(op) != "Conv2d":
            continue
        origins = [adapted_by_id[identity] for identity in op["origin_ops_id"]
                   if identity in adapted_by_id and kind(adapted_by_id[identity]) == "Conv2d"]
        require(len(origins) == 1, "Generated convolution has ambiguous adapted origin")
        source = origins[0]
        output_id = source["outputs"][0]["v_id"]
        reference = quantized_by_output[output_id]
        inp, weights = reference["inputs"][:2]
        out = reference["outputs"][0]
        require(inp["dtype"]["layout"] == out["dtype"]["layout"] == "@layout(NHWC)" and
                weights["dtype"]["layout"] == "@layout(HWIO)", "Unsupported logical convolution layout")
        n, h, w, channels = out["dtype"]["shape"]
        kh, kw, per_group, output_channels = weights["dtype"]["shape"]
        require(channels == output_channels and inp["dtype"]["shape"][-1] == per_group * reference["groups"],
                "Convolution channel/group contract differs")
        flops = 2 * n * h * w * channels * kh * kw * per_group
        unique = sum(tensor_bytes(v) for v in reference["inputs"] + reference["outputs"])
        # 融合 SiLU 的循环日志使用融合后的输出 ID；日志可能只包含首次生成的形状。
        # 缺少日志时保留未知，不能从相同形状推定缓存命中或实际调度完全相同。
        loop_rows = loops.get(("Conv2dNode", op["outputs"][0]["v_id"]), [])
        flattened = {key: value for row in loop_rows for key, value in row.items()}
        temporal = {key: value for key, value in flattened.items() if key.endswith("_MPE_t")}
        measured = timing.get(op["op_id"])
        rows.append(dict(operation=op["op_id"], adapted_operation=source["op_id"], output_value=output_id,
                         input_shape=inp["dtype"]["shape"], output_shape=out["dtype"]["shape"],
                         kernel=[kh, kw], groups=reference["groups"], flops=flops,
                         unique_logical_bytes=unique, ideal_flops_per_unique_byte=flops / unique,
                         physical_io_bytes=sum(tensor_bytes(v) for v in op["inputs"] + op["outputs"]),
                         packed_parameter_bytes=sum(tensor_bytes(v) for v in op["params"]),
                         temporal_loop_bounds=temporal, compiler_loops=loop_rows, loop_log_available=bool(loop_rows),
                         mean_ms=measured,
                         effective_gflops=flops / (measured * 1e6) if measured else None,
                         # 这是循环形状提示；只有真实事务/指令分析才能证明对应数据是否重读。
                         examine_input_reuse=temporal.get("kt_MPE_t", 1) > 1 if loop_rows else None,
                         examine_weight_reuse=(temporal.get("pt_MPE_t", 1) * temporal.get("qt_MPE_t", 1) > 1)
                         if loop_rows else None,
                         examine_partial_sums=temporal.get("ct_MPE_t", 1) > 1 if loop_rows else None))
    return sorted(rows, key=lambda row: row["mean_ms"] if row["mean_ms"] is not None else row["flops"], reverse=True)


def copy_groups(graph, timing):
    ops = graph["ops"]
    producers = {v["v_id"]: op for op in ops for v in op["outputs"]}
    consumers = defaultdict(list)
    for op in ops:
        for value in op["inputs"]:
            consumers[value["v_id"]].append(op)
    grouped = defaultdict(list)
    for op in ops:
        if kind(op) != "Copy":
            continue
        require(len(op["inputs"]) == len(op["outputs"]) == 1, "Non-unary Copy needs separate analysis")
        source = producers[op["inputs"][0]["v_id"]]
        targets = consumers[op["outputs"][0]["v_id"]]
        if kind(source) == "Split" and len(targets) == 1 and kind(targets[0]) == "Concat":
            grouped[(source["op_id"], targets[0]["op_id"])].append(op)
    by_id = {op["op_id"]: op for op in ops}
    rows = []
    for (split_id, concat_id), copies in grouped.items():
        split, concat = by_id[split_id], by_id[concat_id]
        src_chunk, src_offset = chunk(split["inputs"][0])
        dst_chunk, dst_offset = chunk(concat["outputs"][0])
        prefix_bytes = tensor_bytes(split["inputs"][0])
        errors = []
        copied = {op["inputs"][0]["v_id"] for op in copies}
        if copied != {v["v_id"] for v in split["outputs"]}:
            errors.append("not_all_split_outputs_copied")
        if split["layer_count"] or concat["layer_count"]:
            errors.append("split_or_concat_has_instructions")
        if dst_offset != 0 or src_offset != 0:
            errors.append("nonzero_parent_offset")
        for op in copies:
            source, destination = op["inputs"][0], op["outputs"][0]
            source_location, destination_location = chunk(source), chunk(destination)
            if source["dtype"] != destination["dtype"] or source_location[0] != src_chunk or \
                    destination_location != (dst_chunk, source_location[1]):
                errors.append("copy_does_not_preserve_prefix_layout")
        # 查找目标前缀的所有实际写入者。Concat/Split/Reshape 零指令视图只描述别名。
        prefix_writers = []
        for op in ops:
            if not op.get("layer_count", 0):
                continue
            for value in op["outputs"]:
                location = chunk(value)
                if location[0] == dst_chunk and location[1] < prefix_bytes:
                    prefix_writers.append(op["op_id"])
        if set(prefix_writers) != {op["op_id"] for op in copies}:
            errors.append("destination_prefix_has_other_writers")
        io_bytes = sum(tensor_bytes(v) for op in copies for v in op["inputs"] + op["outputs"])
        rows.append(dict(split=split_id, concat=concat_id, copies=[op["op_id"] for op in copies],
                         source_chunk=src_chunk, destination_chunk=dst_chunk, prefix_bytes=prefix_bytes,
                         concat_bytes=tensor_bytes(concat["outputs"][0]), logical_copy_io_bytes=io_bytes,
                         mean_copy_ms=sum(timing[op["op_id"]] for op in copies)
                         if all(op["op_id"] in timing for op in copies) else None,
                         prefix_writers=prefix_writers, candidate=not errors, exclusion_reasons=errors,
                         producer_direct_write_proven=False, transformed_graph_validated=False))
    return sorted(rows, key=lambda row: row["mean_copy_ms"] if row["mean_copy_ms"] is not None
                  else row["logical_copy_io_bytes"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--timing-report", type=Path)
    parser.add_argument("--strategy", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {stage: args.model / ("yolo26l_" + stage + ".json") for stage in ("ZG", "adapted", "quantized")}
    graphs = {stage: read_json(path) for stage, path in paths.items()}
    log = args.model / "logs/generate/yolo26l/yolo26l_debug_zg.log"
    timing = {}
    if args.timing_report:
        reports = read_json(args.timing_report)["runs"]
        selected = [run for run in reports if run["strategy"] == args.strategy]
        require(len(selected) == 1 and selected[0]["timer_semantics"] != "unverified", "Verified timer report required")
        require(selected[0]["identities"]["model.json"] == digest(paths["ZG"]), "Timing belongs to another model")
        timing = {row["operation"]: row["milliseconds"]["mean"] for row in selected[0]["ranked_operations"]}
    result = dict(schema=1, identities={stage: digest(path) for stage, path in paths.items()},
                  codegen_log_sha256=digest(log), actual_ddr_bytes_measured=False, graph_modified=False,
                  copy_groups=copy_groups(graphs["ZG"], timing),
                  convolutions=convolution_rows(graphs["ZG"], graphs["adapted"], graphs["quantized"],
                                               parse_loops(log.read_text(encoding="utf-8")), timing))
    result["summary"] = dict(convolutions=len(result["convolutions"]),
                              convolution_flops=sum(row["flops"] for row in result["convolutions"]),
                              copy_groups=len(result["copy_groups"]),
                              prefix_candidates=sum(row["candidate"] for row in result["copy_groups"]))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
