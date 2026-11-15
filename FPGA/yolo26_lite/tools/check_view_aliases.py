"""核对零指令视图的生成地址关系与 apply 后公开物理放置，不启动 SDK。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_layout import chunk, kind
from analyze_npu import read_json, require, tensor_bytes


def view_constraints(graph):
    result = []
    for op in graph["ops"]:
        if not op["_type_key"].endswith("HardOp") or op.get("layer_count", 0) != 0:
            continue
        groups = defaultdict(dict)
        for value in op["inputs"] + op["outputs"]:
            owner, offset = chunk(value)
            groups[owner][value["v_id"]] = offset
        if kind(op) in ("Concat", "Split", "Reshape"):
            require(len(groups) == 1, "Zero-instruction " + kind(op) + " has disconnected chunks: " + str(op["op_id"]))
        if kind(op) == "Reshape":
            require(len(op["inputs"]) == len(op["outputs"]) == 1, "Expected one-input/output reshape")
            source, destination = op["inputs"][0], op["outputs"][0]
            require(chunk(source) == chunk(destination) and tensor_bytes(source) == tensor_bytes(destination),
                    "Zero-instruction reshape changes offset or storage size")
        if kind(op) in ("Concat", "Split"):
            # 当前零指令图使用连续物理片段；非连续映射需要独立证明，不按同一 chunk 放行。
            combining = kind(op) == "Concat"
            whole = op["outputs"] if combining else op["inputs"]
            parts = op["inputs"] if combining else op["outputs"]
            require(len(whole) == 1 and parts, "Expected a complete contiguous view")
            cursor = chunk(whole[0])[1]
            for value in parts:
                require(chunk(value)[1] == cursor, "Zero-instruction view has a gap or reordered part")
                cursor += tensor_bytes(value)
            require(cursor == chunk(whole[0])[1] + tensor_bytes(whole[0]), "Zero-instruction view size differs")
        for owner, values in groups.items():
            if len(values) > 1:
                result.append(dict(operation=op["op_id"], kind=kind(op), chunk=owner, values=values))
    return result


def check_views(graph, deployment):
    backends = deployment["backends"]
    require(len(backends) == 1, "Expected one NPU backend")
    actual = {v["value_id"]: v for v in backends[0]["values"]}
    errors = []
    constraints = view_constraints(graph)
    for group in constraints:
        missing = set(group["values"]) - set(actual)
        require(not missing, "Runtime view values are missing")
        # 同一生成 chunk 的零指令视图必须具有相同物理基址与存储域；仅看 real_to 名称不够。
        addresses = [{"value_id": identity, "compiled_offset": offset,
                      "actual_address": actual[identity]["physical_address"],
                      "base": actual[identity]["physical_address"] - offset,
                      "is_ocm": actual[identity]["is_ocm"]} for identity, offset in group["values"].items()]
        if len({(row["base"], row["is_ocm"]) for row in addresses}) != 1:
            errors.append(dict(group, addresses=addresses))
    return dict(view_groups=len(constraints), consistent=not errors, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_views(read_json(args.graph), read_json(args.deployment))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("view groups:", result["view_groups"], "inconsistent:", len(result["errors"]))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
