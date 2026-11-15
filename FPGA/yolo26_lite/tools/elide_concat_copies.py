"""为连续 Split→Copy→Concat 前缀创建隔离候选；重新生成指令后独立检查物理别名。"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from analyze_layout import chunk, copy_groups, kind
from analyze_npu import read_json, require, tensor_bytes
from common import run_logged, save_json, sha256
from compile_icraft import review_graph
from check_view_aliases import view_constraints


def select_group(generated, adapted, split_id):
    """仅处理整个 Split 按原顺序组成 Concat 前缀的情况；不猜测非连续切片。"""
    candidates = [row for row in copy_groups(generated, {}) if row["split"] == split_id]
    require(len(candidates) == 1 and candidates[0]["candidate"], "Not a qualified prefix candidate")
    group = deepcopy(candidates[0])
    generated_ops = {op["op_id"]: op for op in generated["ops"]}
    adapted_ops = {op["op_id"]: op for op in adapted["ops"]}

    def origin(identity, expected_kind):
        ids = generated_ops[identity]["origin_ops_id"]
        require(len(ids) == 1 and ids[0] in adapted_ops, "Ambiguous adapted origin")
        op = adapted_ops[ids[0]]
        require(kind(op) == expected_kind, "Adapted origin type differs")
        return op

    split = origin(group["split"], "Split")
    concat = origin(group["concat"], "Concat")
    copies = [origin(identity, "Copy") for identity in group["copies"]]
    require(split["axis"] == concat["axis"] == 1, "Only packed channel-axis prefix is qualified")
    parent = split["inputs"][0]
    require(parent["dtype"]["layout"] == "@layout(NCHWc16)" and
            parent["dtype"]["shape"][0] == 1, "Expected batch-one contiguous packed channel split")
    source_ids = [v["v_id"] for v in split["outputs"]]
    by_source = {op["inputs"][0]["v_id"]: op for op in copies}
    require(set(source_ids) == set(by_source), "Copies do not cover the whole split")
    copies = [by_source[identity] for identity in source_ids]
    destination_ids = [op["outputs"][0]["v_id"] for op in copies]
    require([v["v_id"] for v in concat["inputs"][:len(copies)]] == destination_ids,
            "Split order is not the concat prefix order")
    consumers = defaultdict(list)
    for op in adapted["ops"]:
        for value in op["inputs"]:
            consumers[value["v_id"]].append(op["op_id"])
    for op in copies:
        require(op["inputs"][0]["dtype"] == op["outputs"][0]["dtype"], "Copy changes dtype")
        require(consumers[op["outputs"][0]["v_id"]] == [concat["op_id"]],
                "Copy output has another consumer")
    group.update(adapted_split=split["op_id"], adapted_concat=concat["op_id"],
                 adapted_copies=[op["op_id"] for op in copies], parent_value=parent["v_id"],
                 source_values=source_ids, removed_values=destination_ids,
                 concat_output=concat["outputs"][0]["v_id"])
    return group


def transform(generated, adapted, split_ids, fold_parent=False, materialize_branch=False):
    require(split_ids and len(split_ids) == len(set(split_ids)), "Select distinct split IDs")
    require(not materialize_branch or fold_parent, "Materialized branch requires a folded parent prefix")
    groups = [select_group(generated, adapted, identity) for identity in split_ids]
    removed = {identity for group in groups for identity in group["adapted_copies"]}
    replacements = {}
    by_id = {op["op_id"]: op for op in adapted["ops"]}
    for identity in removed:
        op = by_id[identity]
        replacements[op["outputs"][0]["v_id"]] = op["inputs"][0]
    candidate = deepcopy(adapted)
    candidate["ops"] = [op for op in candidate["ops"] if op["op_id"] not in removed]
    # 只更换上述唯一 Concat 消费者；其余数值、运算顺序及 RAW 参数完全不变。
    for op in candidate["ops"]:
        op["inputs"] = [deepcopy(replacements.get(value["v_id"], value)) for value in op["inputs"]]
    for group in groups:
        group["concat_prefix_values"] = group["source_values"]
        if fold_parent:
            # Concat(Split(x)[0], Split(x)[1], ...) 的完整有序前缀等于 x。
            # 以父张量表达前缀，避免 SDK 对同一父存储的多个子别名重复归并。
            concat = next(op for op in candidate["ops"] if op["op_id"] == group["adapted_concat"])
            parent = deepcopy(by_id[group["adapted_split"]]["inputs"][0])
            concat["inputs"] = [parent] + concat["inputs"][len(group["source_values"]):]
            group["concat_prefix_values"] = [group["parent_value"]]
        if materialize_branch:
            # 保留计算分支的一份显式 Slice，避免 Split 与 Concat 形成嵌套的存储归并。
            # 未被任何剩余算子消费的直通半区不再产生独立 Value；Concat 使用完整父张量。
            split = next(op for op in candidate["ops"] if op["op_id"] == group["adapted_split"])
            consumed = {v["v_id"] for op in candidate["ops"] for v in op["inputs"]}
            outputs = [v for v in split["outputs"] if v["v_id"] in consumed]
            require(len(outputs) == 1, "Materialization expects exactly one live computation branch")
            selected = outputs[0]
            begin = [0] * len(parent["dtype"]["shape"])
            axis = split["axis"]
            begin[axis] = sum(v["dtype"]["shape"][axis] for v in split["outputs"]
                              if group["source_values"].index(v["v_id"]) < group["source_values"].index(selected["v_id"]))
            end = list(parent["dtype"]["shape"])
            end[axis] = begin[axis] + selected["dtype"]["shape"][axis]
            split["_type_key"] = "icraft::xir::Slice"
            split["outputs"] = [selected]
            split.pop("axis")
            split.pop("sections")
            split.update(begin=begin, end=end, stride=[1] * len(begin))
            group["materialized_value"] = selected["v_id"]
    return candidate, groups


def verify_storage(graph, groups):
    """检查生成后的上游输出真的别名到目标前缀，且该区域没有其它指令写入者。"""
    producers = {v["v_id"]: op for op in graph["ops"] for v in op["outputs"]}
    values = {v["v_id"]: v for op in graph["ops"] for v in op["outputs"]}
    results = []
    for group in groups:
        parent = values[group["parent_value"]]
        output = values[group["concat_output"]]
        destination, base = chunk(output)
        require(chunk(parent) == (destination, base), "Producer did not move into concat prefix")
        materialized = group.get("materialized_value")
        split = producers[materialized if materialized is not None else group["source_values"][0]]
        concat = producers[group["concat_output"]]
        require(kind(concat) == "Concat" and concat["layer_count"] == 0, "Concat became executable")
        require((materialized is not None and kind(split) == "Slice" and split["layer_count"] > 0) or
                (materialized is None and kind(split) == "Split" and split["layer_count"] == 0),
                "Branch storage implementation differs")
        cursor = base
        if materialized is None:
            for identity in group["source_values"]:
                require(chunk(values[identity]) == (destination, cursor), "Split offsets are not contiguous")
                cursor += tensor_bytes(values[identity])
        else:
            cursor += tensor_bytes(parent)
        require(cursor - base == tensor_bytes(parent), "Split prefix size differs")
        prefix_values = group.get("concat_prefix_values", group["source_values"])
        require([v["v_id"] for v in concat["inputs"][:len(prefix_values)]] ==
                prefix_values, "Generated concat prefix changed")
        cursor_all = base
        for value in concat["inputs"]:
            require(chunk(value) == (destination, cursor_all), "Concat suffix alias differs")
            cursor_all += tensor_bytes(value)
        require(cursor_all - base == tensor_bytes(output), "Concat complete size differs")
        require(not set(group["removed_values"]).intersection(values), "Deleted copy output survives")
        writers = []
        for op in graph["ops"]:
            if not op.get("layer_count", 0):
                continue
            for value in op["outputs"]:
                owner, offset = chunk(value)
                if owner == destination and offset < cursor and offset + tensor_bytes(value) > base:
                    writers.append(op["op_id"])
        producer = producers[group["parent_value"]]
        require(writers == [producer["op_id"]], "Concat prefix has additional instruction writers")
        results.append(dict(parent_value=parent["v_id"], concat_output=output["v_id"],
                            destination_chunk=destination, base_offset=base, prefix_bytes=cursor - base,
                            producer=producer["op_id"], prefix_writers=writers,
                            static_storage_pass=True, numerical_pass=False, board_pass=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=int, action="append", required=True, help="Original generated Split ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true", help="Also run native codegen; never deploy")
    parser.add_argument("--dump-instlogs", action="store_true", help="Request native instruction logs for Tile analysis")
    parser.add_argument("--fold-parent", action="store_true", help="Fold the whole ordered prefix into its parent value")
    parser.add_argument("--materialize-branch", action="store_true", help="Keep one explicit Slice for the used computation branch")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    source = args.model.resolve()
    output = args.output.resolve()
    paths = {stage: source / ("yolo26l_" + stage + ".json") for stage in ("ZG", "adapted")}
    for path in paths.values():
        review_graph(path)
    candidate, groups = transform(read_json(paths["ZG"]), read_json(paths["adapted"]), args.split,
                                  args.fold_parent, args.materialize_branch)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "yolo26l_adapted.json", candidate)
    shutil.copyfile(str(paths["adapted"].with_suffix(".raw")), str(output / "yolo26l_adapted.raw"))
    manifest = dict(schema=1, transformation="complete_split_concat_prefix_copy_elision", groups=groups,
                    source={stage: sha256(path) for stage, path in paths.items()},
                    source_raw_sha256=sha256(paths["adapted"].with_suffix(".raw")),
                    candidate_adapted_sha256=sha256(output / "yolo26l_adapted.json"), board_access=False)
    save_json(output / "candidate.json", manifest)
    # 不重复 adapt：CopyFtmpPass 会重新引入复制。由 generate 重新规划存储及编码指令，
    # 不修改已生成 RAW 中的指令地址，也不假定运行时物理地址可手工互换。
    config = '\n'.join(['[generate]', 'json = "./yolo26l_adapted.json"', 'raw = "./yolo26l_adapted.raw"',
                        'jr_path = "./"', 'log_path = "./logs/generate"', 'rows = 4', 'cols = 4', 'cores = 1', ''])
    if args.dump_instlogs:
        config += "dump_instlogs = true\n"
    (output / "candidate.toml").write_text(config, encoding="utf-8")
    if not args.generate:
        print("Prepared candidate only:", output)
        return
    runtime = read_json(args.runtime_manifest)
    search = [str(args.bin.resolve())]
    for component in runtime["components"]:
        search.extend(component["bin_dirs"])
        for item in component["dlls"]:
            require(sha256(item["path"]) == item["sha256"], "Compiler runtime DLL identity differs")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(search) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
    run = run_logged([args.bin / "icraft.exe", "generate", "candidate.toml"], output, "generate", env)
    graph_path = output / "yolo26l_ZG.json"
    review = review_graph(graph_path)
    save_json(output / "generate.review.json", dict(run=run, graph=review))
    require(not review["host_compute"], "Candidate introduced host compute")
    generated = read_json(graph_path)
    original = read_json(paths["ZG"])
    deleted = sum(len(group["copies"]) for group in groups)
    require(len(generated["ops"]) == len(original["ops"]) - deleted, "Unexpected generated op count")
    view_constraints(generated)
    save_json(output / "storage-proof.json", verify_storage(generated, groups))
    print("Generated candidate; static storage passed; numerical and board validation pending:", output)


if __name__ == "__main__":
    main()
