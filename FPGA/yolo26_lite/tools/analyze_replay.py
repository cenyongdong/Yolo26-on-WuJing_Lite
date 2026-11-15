"""复核已关闭的板端文件序列。读取所有完整参考张量，保存计时和身份。"""
import argparse
from array import array
import math
from pathlib import Path
import sys

from analyze_npu import read_json
from common import save_json
from probe_contract import digest, model_geometry, require, validate_sequence_completion
from model_geometry import ModelGeometry


def analyze(root):
    manifest = read_json(root / "contract.json")
    completion = read_json(root / "native-complete.json")
    outcome = read_json(root / "outcome.json")
    summary = read_json(root / "native-sequence-summary.json")
    preflight = read_json(root / "preflight.json")
    initialized = read_json(root / "native-initialized.json")
    permit = read_json(root / "startup-permit.json")
    boot = preflight["boot_id"]
    require(manifest["source"] == permit["source"] == outcome["source"] == "file_sequence" and
            outcome["boot_id"] == permit["boot_id"] == boot and outcome["device_owners"] == [],
            "Sequence has no verified native shutdown")
    require(preflight["boot_file"]["sha256"] == manifest["boot_sha256"] and
            preflight["contract_sha256"] == digest(root / "contract.json") and
            permit["duration_seconds"] == manifest["duration_seconds"],
            "BOOT, contract or duration differs from the launch proof")
    timed = validate_sequence_completion(completion, boot, manifest, outcome["native_exit"], summary)
    require(timed == outcome["completed"], "Supervisor counted different completions")
    require(outcome["native_shutdown_complete"] is True and outcome["allocation_count_after"] == 0,
            "Supervisor closure differs from native closure")
    size = model_geometry(manifest["model_variant"])
    geometry = ModelGeometry(*size)
    require((initialized["input_width"], initialized["input_height"]) == size and
            initialized["output_anchors"] == geometry.anchors, "Native tensor geometry differs")
    checked = int(summary["warmup_frames"])
    output_ids = []
    for slot in range(checked):
        path = root / ("native-sequence-input-{}.f32".format(slot))
        values = array("f")
        require(values.itemsize == 4 and path.stat().st_size == 4 * 84 * geometry.anchors,
                "Archived tensor length differs")
        values.frombytes(path.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        require(all(math.isfinite(value) for value in values), "Archived tensor contains a nonfinite value")
        output_ids.append(digest(path))
    totals = summary["stage_totals_ns"]
    fields = {"clear_ns", "final_wait_ns", "forward_ns", "input_write_ns", "normalize_ns",
              "output_read_ns", "postprocess_ns"}
    require(set(totals) == fields and all(type(value) is int and value >= 0 for value in totals.values()),
            "Invalid stage totals")
    metrics = summary["metrics"]
    inference = metrics["stages"]["inference"]
    slow_calls = summary.get("longest_calls", [])
    if "longest_calls" in summary:
        count = summary["slow_calls_at_least_40ms"]
        require(type(count) is int and count >= 0 and len(slow_calls) == min(16, count),
                "Bounded slow-call count differs")
        for row in slow_calls:
            start, end = row["start_ns"], row["complete_ns"]
            require(metrics["start_ns"] <= start < end and end - start == row["wall_ns"] >= 40000000 and
                    set(row["stages"]) == fields and
                    all(type(value) is int and value >= 0 for value in row["stages"].values()),
                    "Slow-call record differs from its timing boundaries")
        elapsed = [row["wall_ns"] for row in slow_calls]
        require(elapsed == sorted(elapsed, reverse=True), "Slow-call records are out of order")
    return dict(geometry=list(size), anchors=geometry.anchors, boot_id=boot, boot_sha256=manifest["boot_sha256"],
                identities=manifest["files"], sdk_libraries=preflight["libraries"],
                completed=timed, native_shutdown_complete=True, allocation_count_after=0,
                input_transfer=initialized["input_transfer"], completion_layers=initialized["completion_layers"],
                normalization_backend=initialized.get("normalization_backend"),
                distinct_rgb_frames=manifest["replay_frames"], output_sha256=output_ids,
                distinct_complete_outputs=len(set(output_ids)), stage_totals_ns=totals,
                stage_mean_ms={name: value / timed / 1000000 for name, value in totals.items()} if timed else {},
                inference=inference, measurement_complete=metrics["measurement_complete"],
                every_inference_second_at_least_30=metrics["every_inference_second_at_least_30"],
                camera_30fps_claim=False, seconds=manifest["duration_seconds"],
                full_tensor_checks_during_measurement=summary["checks_during_measurement"],
                full_tensor_checks_after_measurement=summary["checked_after_measurement"],
                slow_calls_at_least_40ms=summary.get("slow_calls_at_least_40ms"), longest_calls=slow_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve the previous sequence analysis")
    result = analyze(args.run)
    save_json(args.output, result)
    fps = result["inference"]["mean_fps"]
    worst = min(result["inference"]["windows"])
    print("Completed: {} | mean {:.3f} FPS | minimum 1 s count: {}".format(result["completed"], fps, worst))


if __name__ == "__main__":
    main()
