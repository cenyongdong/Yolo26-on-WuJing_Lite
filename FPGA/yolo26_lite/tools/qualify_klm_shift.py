"""离线核验 KLM 平移只改变目标指令；保存完整生成模拟输出的逐位比较身份。"""
import argparse
import json
import math
import os
from pathlib import Path
import struct

from common import save_json, sha256
from check_view_aliases import view_constraints
from split_conv_channels import parameters, parameter_bytes
from probe_contract import require


def simulation_pair(reference, candidate, reference_hash, candidate_hash):
    receipts = [json.loads((path / "simulation.json").read_text()) for path in (reference, candidate)]
    for receipt, graph_hash in zip(receipts, (reference_hash, candidate_hash)):
        require(receipt["stage"] == "ZG" and receipt["run"]["native_exit"] == 0 and
                receipt["graph_sha256"] == graph_hash and receipt["output_value_ids"] == [2301],
                "Simulation does not belong to the generated graph")
    require(receipts[0]["case"] == receipts[1]["case"] and
            receipts[0]["encoded_input_sha256"] == receipts[1]["encoded_input_sha256"],
            "Simulation inputs differ")
    outputs = [path / "native/yolo26l/ftmpSFB/2301.ftmp" for path in (reference, candidate)]
    old, new = [path.read_bytes() for path in outputs]
    require(len(old) == 705600 * 4 and old == new, "Complete generated output differs")
    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", new)), "Nonfinite output")
    return dict(case=receipts[0]["case"], values=705600, bitwise_equal=True,
                input_sha256=receipts[0]["encoded_input_sha256"],
                reference_sha256=sha256(outputs[0]), actual_sha256=sha256(outputs[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-simulation", required=True, type=Path, action="append")
    parser.add_argument("--candidate-simulation", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous qualification")
    roots = [args.reference.resolve(strict=True), args.candidate.resolve(strict=True)]
    graphs = [json.loads((root / "yolo26l_ZG.json").read_text()) for root in roots]
    require(graphs[0]["ops"] == graphs[1]["ops"], "Operator descriptions or storage relations changed")
    views = view_constraints(graphs[1])
    handle = os.add_dll_directory(str(args.bin.resolve(strict=True)))
    from icraft import xir
    networks = []
    for root in roots:
        network = xir.Network.CreateFromJsonFile(str(root / "yolo26l_ZG.json"))
        network.loadParamsFromFile(str(root / "yolo26l_ZG.raw"))
        networks.append(network)
    old, new = [parameters(network, xir) for network in networks]
    require(set(old) == set(new), "Parameter identities changed")
    differences = [identity for identity in old if parameter_bytes(old[identity], xir) != parameter_bytes(new[identity], xir)]
    require(differences == [8863], "Changes extend beyond the #3405 instruction block")
    hashes = [sha256(root / "yolo26l_ZG.json") for root in roots]
    require(len(args.reference_simulation) == len(args.candidate_simulation) == 3, "Need three complete input cases")
    cases = [simulation_pair(left, right, *hashes)
             for left, right in zip(args.reference_simulation, args.candidate_simulation)]
    require(sorted(case["case"] for case in cases) == [0, 1, 2], "Missing or duplicated input case")
    report = dict(schema=1, offline_qualified=True, board_tested=False, parameter_objects=len(old),
                  changed_parameter_ids=differences, mathematical_parameter_bytes_unchanged=True,
                  operator_descriptions_equal=True, view_groups=len(views), complete_output_cases=cases,
                  graph_sha256=hashes[1], raw_sha256=sha256(roots[1] / "yolo26l_ZG.raw"),
                  instruction_change="KLM base +256 U32 words; no task reordering or prefetch yet")
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
