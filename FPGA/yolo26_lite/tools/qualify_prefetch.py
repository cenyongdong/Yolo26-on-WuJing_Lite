"""完整验证固定跨层预取的数学参数、外部存储绑定、全部 KLM 访问与三组指令模拟输出。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from analyze_klm_handoff import parse_tasks
from check_view_aliases import view_constraints
from common import save_json, sha256
from probe_contract import require
from qualify_klm_shift import simulation_pair
from split_conv_channels import parameter_bytes, parameters


INITIAL = {3, 43, 83, 123}
SHIFT_BYTES = 8192


def address_handoff(old_first, old_second, new_first, new_second, position):
    """TaskGen 整数地址只用于范围证明，不从舍入文本重建任何数学常数。"""
    inserted = set(range(position, position + 4))
    prefetch = [row for row in new_first if row["task"] in inserted]
    retained = [deepcopy(row) for row in new_first if row["task"] not in inserted]
    require(len(prefetch) == 8 and all(row["write"] and row["task_type"] == "BoardcastTask" for row in prefetch),
            "Four complete two-MPE prefetch broadcasts are required")
    for row in retained:
        if row["task"] >= position + 4:
            row["task"] -= 4
    require(retained == old_first, "Earlier mathematical KLM accesses changed")
    require(max(row["end"] for row in old_first) <= SHIFT_BYTES, "Earlier KLM range overlaps prefetch")
    initial = [row for row in old_second if row["write"] and row["task"] in INITIAL]
    require(len(initial) == 8 and len({row["mpe"] for row in initial}) == 8, "Initial MPE coverage changed")
    keyed = {tuple(row["mpe"]): row for row in prefetch}
    require(len(keyed) == 8, "Prefetch duplicated or omitted an MPE")
    for original in initial:
        actual = keyed[tuple(original["mpe"])]
        require(actual["begin"] == original["begin"] + SHIFT_BYTES and
                actual["end"] == original["end"] + SHIFT_BYTES and
                actual["payload_bytes"] == original["payload_bytes"], "Prefetched block differs from first consumer")
    require(len(new_second) == len(old_second), "KLM access coverage changed")
    primes = []
    for original, actual in zip(old_second, new_second):
        expected = deepcopy(original)
        if original["write"] and original["task"] in INITIAL:
            expected.update(begin=0, end=32, payload_bytes=32)
            primes.append(actual)
        else:
            expected.update(begin=original["begin"] + SHIFT_BYTES, end=original["end"] + SHIFT_BYTES)
        require(actual == expected, "Inconsistent KLM relocation or state-establishing write")
    require(all(row["begin"] >= SHIFT_BYTES for row in new_second if not row["write"]),
            "Mathematical task reads the scratch state-establishing word")
    maximum = max(row["end"] for row in new_second)
    require(maximum <= 65536, "KLM capacity exceeded")
    return dict(previous_accesses=len(old_first), following_accesses=len(old_second), mpes=len(keyed),
                previous_math_accesses_unchanged=True, all_following_accesses_checked=True,
                previous_end_bytes=max(row["end"] for row in old_first),
                prefetch_begin_bytes=min(row["begin"] for row in prefetch),
                prefetch_end_bytes=max(row["end"] for row in prefetch), following_end_bytes=maximum,
                scratch_interval=[0, 32], scratch_read_by_math=False,
                logical_prefetch_source_bytes=sum(row["payload_bytes"] for row in prefetch) // 2,
                logical_prime_source_bytes=sum(row["payload_bytes"] for row in primes) // 2,
                physical_DDR_bytes=None, hardware_overlap_verified=False)


def graph_boundaries(old_graph, new_graph):
    old, new = [{operation["op_id"]: operation for operation in graph["ops"]} for graph in (old_graph, new_graph)]
    require(old.keys() == new.keys(), "Operator identities changed")
    changed = []
    for identity in old:
        if old[identity] == new[identity]:
            continue
        require(identity in (3404, 3405), "Unrelated operator changed")
        a, b = deepcopy(old[identity]), new[identity]
        require(len(a["instr"]) == len(b["instr"]) == 1, "Unexpected instruction block count")
        a["instr"][0]["dtype"]["shape"] = b["instr"][0]["dtype"]["shape"]
        if identity == 3404:
            require(b["params"][-1] == new[3405]["params"][0], "Prefetch does not share the next kernel binding")
            a["params"].append(deepcopy(b["params"][-1]))
        require(a == b, "Mathematical boundary or storage relationship changed")
        changed.append(identity)
    require(changed == [3404, 3405], "Expected exactly the adjacent target pair")
    return len(view_constraints(new_graph))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-logs", type=Path, required=True)
    parser.add_argument("--reference-simulation", type=Path, action="append", required=True)
    parser.add_argument("--candidate-simulation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous qualification evidence")
    roots = [args.reference.resolve(strict=True), args.candidate.resolve(strict=True)]
    generated = json.loads((roots[1] / "prefetch-generation.json").read_text())
    require(generated["restored"] and not generated["keep_original_loads"], "Not a completed prefetch candidate")
    graphs = [json.loads((root / "yolo26l_ZG.json").read_text()) for root in roots]
    view_groups = graph_boundaries(*graphs)
    handle = os.add_dll_directory(str(args.bin.resolve(strict=True)))
    from icraft import xir
    networks = []
    for root in roots:
        network = xir.Network.CreateFromJsonFile(str(root / "yolo26l_ZG.json"))
        network.loadParamsFromFile(str(root / "yolo26l_ZG.raw"))
        networks.append(network)
    old, new = [parameters(network, xir) for network in networks]
    require(old.keys() == new.keys(), "Parameter identities changed")
    differences = [identity for identity in old if parameter_bytes(old[identity], xir) != parameter_bytes(new[identity], xir)]
    require(differences == [8858, 8863], "Parameter changes extend beyond two instruction blocks")
    # 检查每个重复引用，而不只检查 v_id 字典中最后一个对象。
    references = 0
    for operation in networks[1].ops:
        for value in list(operation.inputs) + list(operation.paramsAttrs()):
            if value.isParams():
                parameter = value if isinstance(value, xir.Params) else xir.Params(value)
                require(parameter_bytes(parameter, xir) == parameter_bytes(new[parameter.v_id], xir),
                        "Shared Params references contain different bytes")
                references += 1
    logs = [args.reference_logs, roots[1] / "unbound/logs/yolo26l"]
    access, log_hashes = [], {}
    for directory in logs:
        for operation in (3404, 3405):
            path = directory / ("hardOp%d_zg_insts.log" % operation)
            with path.open() as stream:
                access.append(parse_tasks(stream))
            log_hashes[str(path)] = sha256(path)
    addresses = address_handoff(*access, generated["insert_before"])
    hashes = [sha256(root / "yolo26l_ZG.json") for root in roots]
    require(generated["graph_sha256"] == hashes[1] and generated["raw_sha256"] == sha256(roots[1] / "yolo26l_ZG.raw"),
            "Candidate bytes changed after generation")
    require(len(args.reference_simulation) == len(args.candidate_simulation) == 3, "Three complete cases required")
    cases = [simulation_pair(reference, candidate, *hashes) for reference, candidate in
             zip(args.reference_simulation, args.candidate_simulation)]
    require(sorted(row["case"] for row in cases) == [0, 1, 2], "Missing or repeated simulation case")
    report = dict(schema=1, offline_qualified=True, board_tested=False, parameter_objects=len(old),
                  parameter_references=references, changed_instruction_ids=differences,
                  mathematical_parameter_bytes_unchanged=True, boundary_and_storage_checks=True,
                  view_groups=view_groups, address_handoff=addresses, complete_output_cases=cases,
                  graph_sha256=hashes[1], raw_sha256=generated["raw_sha256"], task_log_sha256=log_hashes,
                  implementation="four next-layer broadcasts; all KLM +8KiB; four 32-byte scratch writes establish state")
    save_json(args.output, report)
    print("Prefetch: complete parameter, address and three-output checks passed; board performance remains unmeasured.")
    return handle


if __name__ == "__main__":
    main()
