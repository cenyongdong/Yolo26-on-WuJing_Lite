"""区分校准数量、校准分区和精度回退的影响；禁止比较不同留出输入或评价协议。"""
import argparse
import json
import math
from pathlib import Path

from analyze_npu import read_json, require
from common import save_json, sha256


PROTOCOL_KEYS = ("images", "labels", "classes", "nms", "max_det", "confidence_filter", "metrics_library")


def check_protocol(reference, actual):
    require(all(reference[key] == actual[key] for key in PROTOCOL_KEYS), "Evaluation protocol differs")
    for key in ("dataset_manifest_sha256", "cases"):
        require(reference["identity"][key] == actual["identity"][key], "Held-out inputs differ")
    require(reference["identity"]["full_held_out_set"] and actual["identity"]["full_held_out_set"],
            "Partial validation cannot be compared as a full result")
    for metric in (reference, actual):
        require(all(math.isfinite(metric[key]) and 0 <= metric[key] <= 1 for key in ("map50", "map50_95")),
                "Invalid AP result")


def check_factor(first, second, factor):
    """原始图和校准池保持相同，按试验目的只放开指定因素。"""
    require(first["source"]["json_sha256"] == second["source"]["json_sha256"] and
            first["source"]["raw_sha256"] == second["source"]["raw_sha256"] and
            first["calibration_manifest_sha256"] == second["calibration_manifest_sha256"],
            "Source graph, weights or calibration pool changed")
    n1, n2 = first["calibration_count"], second["calibration_count"]
    b1, b2 = first["native_batch"], second["native_batch"]
    require(min(n1, n2, b1, b2) > 0 and n1 % b1 == 0 and n2 % b2 == 0, "Invalid calibration partitions")
    p1, p2 = first["precision_policy"], second["precision_policy"]
    if factor == "partition":
        require(n1 == n2 and p1 == p2 and b1 != b2, "Partition experiment changed another factor")
    elif factor == "calibration_count":
        require(n1 != n2 and p1 == p2 and n1 // b1 == n2 // b2,
                "Calibration-size experiment must keep precision and images per partition fixed")
    elif factor == "precision":
        require(n1 == n2 and b1 == b2 and p1 != p2, "Precision experiment changed calibration")
        require([(row["op_id"], row["type"]) for row in p1] == [(row["op_id"], row["type"]) for row in p2],
                "Precision experiment changed operator structure")
    else:
        raise ValueError("Unknown experiment factor")


def load_experiment(model, results):
    identity = read_json(model / "input-identity.json")
    metric = read_json(results / "metrics.json")
    generated = read_json(model / "generate.review.json")
    require(metric["identity"]["graph_sha256"] == sha256(model / "yolo26l_ZG.json") ==
            generated["graph"]["json_sha256"] and metric["identity"]["raw_sha256"] ==
            sha256(model / "yolo26l_ZG.raw") == generated["graph"]["raw_sha256"], "Model identity changed")
    require(not generated["graph"]["host_compute"] and generated["run"]["native_exit"] == 0,
            "Candidate is not a completed NPU graph")
    manifest_path = Path(identity["calibration_root"]) / "manifest.json"
    require(sha256(manifest_path) == identity["calibration_manifest_sha256"] ==
            metric["identity"]["dataset_manifest_sha256"], "Calibration or validation manifest changed")
    require(metric["identity"] == read_json(results / "identity.json"), "Validation identity changed")
    manifest = read_json(manifest_path)
    expected = {row["case"]: row["input_sha256"] for row in manifest["validation"]}
    require(len(metric["identity"]["cases"]) == len(expected) == metric["images"] and
            set(metric["identity"]["cases"]) == set(expected), "Not all held-out images were evaluated once")
    for case in metric["identity"]["cases"]:
        name = "case-{}-dense.npy".format(case)
        receipt = read_json(results / (name + ".json"))
        require(receipt["input_sha256"] == expected[case] and receipt["output_sha256"] == sha256(results / name),
                "Complete numerical output changed")
    return dict(identity=identity, metrics=metric, generated=generated,
                model=str(model.resolve()), results=str(results.resolve()))


def summarize(reference, experiments):
    comparisons = (("partition", "base32", "groups32"),
                   ("calibration_count", "groups32", "cal96"),
                   ("precision", "base32", "psa32"),
                   ("precision", "base32", "stem32"))
    rows = {}
    for name, item in experiments.items():
        metric, identity = item["metrics"], item["identity"]
        check_protocol(reference, metric)
        rows[name] = dict(calibration_count=identity["calibration_count"], native_batch=identity["native_batch"],
                         images_per_partition=identity["calibration_count"] // identity["native_batch"],
                         profile=identity["profile"], map50=metric["map50"], map50_95=metric["map50_95"],
                         delta_vs_fp32_pp=100 * (metric["map50_95"] - reference["map50_95"]),
                         graph_sha256=metric["identity"]["graph_sha256"],
                         within_proxy_one_point_target=metric["map50_95"] >= reference["map50_95"] - .01)
    effects = []
    for factor, before, after in comparisons:
        check_factor(experiments[before]["identity"], experiments[after]["identity"], factor)
        effects.append(dict(factor=factor, before=before, after=after,
                            delta_map50_pp=100 * (rows[after]["map50"] - rows[before]["map50"]),
                            delta_map50_95_pp=100 * (rows[after]["map50_95"] - rows[before]["map50_95"])))
    return dict(schema=1, reference_map50_95=reference["map50_95"], experiments=rows, effects=effects,
                held_out_images=reference["images"], labels=reference["labels"], classes=reference["classes"],
                board_performance_measured=False, official_coco_validation=False,
                limit="Deterministic comparison on a small fixed proxy set; no population accuracy claim or isolated-layer causality from cumulative traces.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="FP32 reference metrics JSON")
    for name in ("base32", "groups32", "cal96", "psa32", "stem32"):
        parser.add_argument("--" + name, type=Path, nargs=2, required=True, metavar=("MODEL", "RESULTS"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiments = {name: load_experiment(*getattr(args, name))
                   for name in ("base32", "groups32", "cal96", "psa32", "stem32")}
    result = summarize(read_json(args.reference), experiments)
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
