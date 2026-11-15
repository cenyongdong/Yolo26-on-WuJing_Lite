"""从同一生成图和已完成板测追踪张量生产／消费、别名和真实搬运，不访问设备。"""
import argparse
from collections import defaultdict
from pathlib import Path

from analyze_layout import chunk, kind
from analyze_npu import read_json, require, tensor_bytes
from analyze_pointwise import region_between
from common import save_json, sha256


def trace(graph, deployment, input_value, output_value):
    result = trace_between(graph, deployment, [input_value], [output_value])
    result.update(input_value=input_value, output_value=output_value)
    return result


def trace_between(graph, deployment, input_values, output_values):
    """共同边界包含共享生产者、拼接及恢复分支，避免漏记转移到上游的搬运。

    多输入算子保留全部操作数；单输入算子额外报告是否完全共享存储。
    地址相同只表示复用缓冲区，不能据此认定数据留在片上或没有 DDR 事务。
    """
    require(len(deployment["backends"]) == 1, "Expected one NPU backend")
    placed = {value["value_id"]: value for value in deployment["backends"][0]["values"]}
    require(len(placed) == len(deployment["backends"][0]["values"]), "Duplicate deployed value identities")
    producers, consumers, values = {}, defaultdict(list), {}
    for op in graph["ops"]:
        for value in op["outputs"]:
            require(value["v_id"] not in producers, "Multiple producers for one value")
            producers[value["v_id"]] = op
            values[value["v_id"]] = value
        for value in op["inputs"]:
            consumers[value["v_id"]].append(op)
    operations = region_between(graph, input_values, output_values)
    identities = {v["v_id"] for op in operations for v in op["inputs"] + op["outputs"]}
    rows = []
    for identity in sorted(identities):
        require(identity in values and identity in placed, "Missing generated or deployed value")
        value, actual = values[identity], placed[identity]
        require(actual["shape"] == value["dtype"]["shape"] and
                actual["storage_bytes"] == tensor_bytes(value), "Deployment shape or storage size differs")
        owner, offset = chunk(value)
        producer = producers[identity]
        rows.append(dict(value_id=identity, shape=actual["shape"], layout=actual["layout"],
            storage_bytes=actual["storage_bytes"], compiled_chunk=owner, compiled_offset=offset,
            physical_address=actual["physical_address"], is_ocm=actual["is_ocm"],
            producer=dict(operation=producer["op_id"], kind=kind(producer),
                          completion_layers=producer.get("layer_count", 0)),
            consumers=[dict(operation=op["op_id"], kind=kind(op)) for op in consumers[identity]]))
    edges = []
    for op in operations:
        edge = dict(operation=op["op_id"], kind=kind(op),
                    input_values=[value["v_id"] for value in op["inputs"]],
                    output_values=[value["v_id"] for value in op["outputs"]],
                    completion_layers=op.get("layer_count", 0))
        edges.append(edge)
        if len(op["inputs"]) != 1 or len(op["outputs"]) != 1:
            continue
        source, destination = (placed[op[key][0]["v_id"]] for key in ("inputs", "outputs"))
        same_storage = (source["physical_address"], source["is_ocm"], source["storage_bytes"]) == (
            destination["physical_address"], destination["is_ocm"], destination["storage_bytes"])
        if kind(op) == "Reshape" and op.get("layer_count") == 0:
            require(same_storage, "Zero-instruction reshape does not preserve its storage")
        edge.update(input_value=source["value_id"], output_value=destination["value_id"],
                    same_physical_storage=same_storage)
    return dict(input_values=list(input_values), output_values=list(output_values), tensors=rows, operations=edges,
                actual_ddr_transactions_measured=False, note="Same physical storage is not on-chip retention; "
                "a DDR-producing operator and its consumer still write/read that storage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="Closed run containing contract and deployment")
    parser.add_argument("--input-value", type=int, action="append", required=True)
    parser.add_argument("--output-value", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous memory-flow evidence")
    identity = sha256(args.graph)
    require(read_json(args.run / "contract.json")["files"]["model.json"] == identity,
            "Deployment belongs to another generated graph")
    outcome = read_json(args.run / "outcome.json")
    require(outcome["native_shutdown_complete"] and outcome["allocation_count_after"] == 0,
            "Read evidence only from a safely closed run")
    deployment = args.run / "npu-deployment.json"
    if len(args.input_value) == len(args.output_value) == 1:
        result = trace(read_json(args.graph), read_json(deployment), args.input_value[0], args.output_value[0])
    else:
        result = trace_between(read_json(args.graph), read_json(deployment), args.input_value, args.output_value)
    result.update(graph_sha256=identity, deployment_sha256=sha256(deployment), board_access=False)
    save_json(args.output, result)
    for row in result["operations"]:
        print(row["operation"], row["kind"], row["input_values"], "->", row["output_values"],
              "same storage:", row.get("same_physical_storage"), "layers:", row["completion_layers"])


if __name__ == "__main__":
    main()
