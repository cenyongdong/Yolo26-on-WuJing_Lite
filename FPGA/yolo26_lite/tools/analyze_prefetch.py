"""比较同程序的基线、KLM 平移及跨层预取；保守连接层进度与实际接口分箱。"""
import argparse
import json
from pathlib import Path

from analyze_bus_timeline import analyze as analyze_bus, FIELDS
from analyze_counter_timeline import analyze as analyze_counters
from analyze_npu import analyze_run, digest, read_json, require, statistics_ms


def transition_bounds(samples, completed_layers):
    """层计数读取不是原子的：只用包含实际读取的主机时间区间给出完成边界。"""
    lower = [row["begin_ns"] for row in samples if row["layer_before"] < completed_layers]
    upper = [row["end_ns"] for row in samples if row["layer_after"] >= completed_layers]
    require(lower and upper, "Target transition is outside the observed progress interval")
    bounds = [max(lower), min(upper)]
    require(bounds[0] <= bounds[1], "Progress transition bounds are reversed")
    return bounds


def region_observation(bus, counters, first_layer, layer_count):
    """不将边界箱按时间比例分摊。下界只计确定完整落入区域的箱，上界计所有可能相交箱。"""
    begin = transition_bounds(counters["samples"], first_layer)
    end = transition_bounds(counters["samples"], first_layer + layer_count)
    require(begin[0] <= end[0] and begin[1] <= end[1], "Reversed operation interval")
    inside, possible = [], []
    for row in bus["timeline"]:
        start_low, start_high = row["begin_host_ns_bounds"]
        end_low, end_high = row["end_host_ns_bounds"]
        require(start_low <= start_high and end_low <= end_high, "Invalid hardware/host time bounds")
        if end_high > begin[0] and start_low < end[1]:
            possible.append(row)
        if start_low >= begin[1] and end_high <= end[0]:
            inside.append(row)
    require(possible, "No captured bin can intersect the target region")
    counts = {name: [sum(row[name] for row in inside), sum(row[name] for row in possible)] for name in FIELDS}
    require(all(low <= high for low, high in counts.values()), "Event bounds are inconsistent")
    return dict(first_layer=first_layer, layer_count=layer_count,
                start_host_ns_bounds=begin, end_host_ns_bounds=end,
                duration_ms_bounds=[max(0, end[0] - begin[1]) / 1e6, (end[1] - begin[0]) / 1e6],
                wholly_inside_bins=[row["index"] for row in inside],
                possibly_intersecting_bins=[row["index"] for row in possible],
                event_count_bounds=counts,
                interface_return_payload_bytes_bounds=[count * 64 for count in counts["r_accepted"]],
                ddr_bytes=None, mpe_wait_ms=None,
                note="Whole-bin bounds only; no uniform-event or exclusive-compute/memory assumption")


def inspect_case(name, graph_path, directory):
    graph = read_json(graph_path)
    report = analyze_run(directory, graph, digest(graph_path), warmup=3, semantics="durations")
    report["name"] = name
    manifest = read_json(directory / "contract.json")
    require(manifest.get("bus_trace") and manifest.get("counter_trace"), "Both diagnostic traces are required")
    operations = {row["operation"]: row for row in report["ranked_operations"]}
    first, second = operations[3404], operations[3405]
    require(first["first_layer"] + first["layer_count"] == second["first_layer"] and
            first["layer_count"] == second["layer_count"] == 1, "Target layers are no longer adjacent")
    traces, combined = [], []
    for frame in range(manifest["iterations"] - manifest["diagnostic_samples"], manifest["iterations"]):
        timers = read_json(directory / ("npu-timers-{}.json".format(frame)))
        frame_record = read_json(directory / ("native-frame-{}.json".format(frame)))
        require((timers["epoch"], timers["frame"]) == (frame_record["epoch"], frame), "Trace frame differs")
        counters = timers["counter_timeline"]
        analyze_counters(counters, timers["expected_layers"])
        bus = analyze_bus(timers["bus_timeline"], counters)
        require(not bus["overflow"], "Capture overflow prevents a full inference timeline")
        regions = {}
        for key, start, count in (("3404", first["first_layer"], 1), ("3405", second["first_layer"], 1),
                                  ("3404+3405", first["first_layer"], 2)):
            regions[key] = region_observation(bus, counters, start, count)
        combined.append(sum(timers["api_layer_ms"][first["first_layer"]:second["first_layer"] + 1]))
        traces.append(dict(frame=frame, bus=bus, regions=regions))
    report["target_region_ms"] = {
        "3404": first["milliseconds"], "3405": second["milliseconds"], "3404+3405": statistics_ms(combined)}
    report["traces"] = traces
    return report


def compare(reports):
    require(len(reports) == 3 and {row["name"] for row in reports} == {"baseline", "shift", "prefetch"},
            "Expected exactly baseline, shift and prefetch")
    baseline = next(row for row in reports if row["name"] == "baseline")
    require(len({row["boot_id"] for row in reports}) == len(reports), "Startup was reused")
    for row in reports:
        require(all(row["identities"][key] == baseline["identities"][key] for key in
                    ("input.rgb", "yolo26_infer_probe")) and row["boot_sha256"] == baseline["boot_sha256"] and
                row["sdk_libraries"] == baseline["sdk_libraries"], "Input, ELF, BOOT or SDK differs")
        require(row["ordinary_frame_range"] == baseline["ordinary_frame_range"] and row["strategy"] == -1,
                "Timing population or OCM policy differs")
        require(row["output_finite"] and row["output_sha256"] == baseline["output_sha256"],
                "Complete finite board outputs differ; preserve results before claiming equivalent speed")
    return [dict(name=row["name"], full_call_ms=row["timings"]["complete_call_ms"],
                 forward_wait_ms=row["timings"]["forward_wait_ms"], target_region_ms=row["target_region_ms"],
                 full_mean_delta_from_baseline_ms=row["timings"]["complete_call_ms"]["mean"] -
                                                 baseline["timings"]["complete_call_ms"]["mean"])
            for row in reports]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", metavar=("NAME", "GRAPH", "RETURNED"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous comparison")
    reports = [inspect_case(name, Path(graph), Path(directory)) for name, graph, directory in args.case]
    comparison = compare(reports)
    result = dict(schema=1, comparison=comparison, runs=reports, all_complete_outputs_bitwise_equal=True,
                  timing_population="First 3 warmups and final diagnostic frames excluded from ordinary calls",
                  statistical_scope="One startup per policy; short-run comparison, not a 30FPS acceptance test",
                  real_DDR_transactions_measured=False, direct_MPE_wait_measured=False, new_board_access=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
