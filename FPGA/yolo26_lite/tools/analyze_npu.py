"""离线核验 OCM 批次、原生计时映射和完整输出；不导入 SDK，不访问设备。"""
import argparse
from array import array
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


# 本轮在板端只读核验了该库的公开 getLayerTime/calcTime 实现，不将格式推广到其它 SDK。
TIMER_XRT_SHA256 = "a29e4eee6106afae0b6aada4a07e85416ac6dfc3e74083033ae5d1411749dc96"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_bytes(value):
    dtype = value["dtype"]
    element = dtype["element_dtype"]
    if isinstance(element, dict):
        element = element["storage_dtype"]
    # Icraft QuantizedType 对有符号 INT8 使用 @sint(8)，按 storage 而非 expressed FP32 计字节。
    match = re.fullmatch(r"@(fp|tf|int|sint|uint)\((\d+)\)", element)
    require(match is not None, "Unknown physical element type: " + str(element))
    shape = dtype["shape"]
    require(all(type(x) is int and x >= 0 for x in shape), "Non-static physical shape")
    bits = math.prod(shape) * int(match.group(2))
    require(bits % 8 == 0, "Non-byte-addressable physical shape")
    return bits // 8


def map_layers(graph, initialized):
    """使用实际融合顺序，而非 op_id 排序。零层节点保留来源，但不占硬件计时槽。"""
    hardops = [op for op in graph["ops"] if op["_type_key"].endswith("::HardOp")]
    by_id = {op["op_id"]: op for op in hardops}
    require(len(by_id) == len(hardops), "Duplicate HardOp identities")
    result, seen, cursor = [], set(), 0
    for group in sorted(initialized["fusion"], key=lambda value: value["first_layer"]):
        require(group["first_layer"] == cursor and group["layer_count"] > 0, "Fusion intervals have gaps/overlap")
        start = cursor
        for identity in group["originals"] or [group["operation"]]:
            require(identity in by_id and identity not in seen, "Fusion membership differs from generated graph")
            seen.add(identity)
            op = by_id[identity]
            layers = op["layer_count"]
            require(type(layers) is int and layers >= 0, "Invalid HardOp layer count")
            kind = op["name"].split("::")[-1]
            result.append(dict(operation=identity, kind=kind[:-4] if kind.endswith("Node") else kind,
                               first_layer=cursor, layer_count=layers, origin_ops=op["origin_ops_id"],
                               input_values=[v["v_id"] for v in op["inputs"]],
                               output_values=[v["v_id"] for v in op["outputs"]],
                               instruction_bytes=sum(tensor_bytes(v) for v in op["instr"]),
                               logical_io_bytes=sum(tensor_bytes(v) for key in ("inputs", "outputs") for v in op[key])))
            cursor += layers
        require(cursor - start == group["layer_count"], "Original layer sum differs from applied fusion")
    require(seen == set(by_id) and cursor == initialized["completion_layers"], "Incomplete native layer coverage")
    return result


def statistics_ms(values):
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "Invalid timing sample")
    ordered = sorted(values)
    # 最近秩法，保留样本数；少量样本的 P99 通常就是最大值，不伪装成长测分位数。
    return dict(samples=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[math.ceil(0.95 * len(values)) - 1],
                p99=ordered[math.ceil(0.99 * len(values)) - 1], maximum=max(values), minimum=min(values))


def layer_durations(sample, expected, semantics):
    values = sample["api_layer_ms"]
    require(sample["expected_layers"] == expected and sample["returned_layer_count"] == len(values) == expected,
            "Native timer count differs from completion layers")
    require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in values),
            "Native timer is non-finite, negative or missing")
    if semantics == "unverified":
        return None
    if semantics == "durations":
        return values
    require(semantics == "cumulative", "Unknown native timer interpretation")
    previous, durations = 0, []
    for value in values:
        require(value >= previous, "Cumulative timer went backwards")
        durations.append(value - previous)
        previous = value
    return durations


def inspect_timer(sample, expected, semantics):
    """原生计时异常不抹掉独立墙钟结果；显式层归因则必须拒绝异常样本。"""
    error = None
    try:
        durations = layer_durations(sample, expected, semantics)
    except ValueError as problem:
        if semantics != "unverified":
            raise
        error, durations = str(problem), None
    values = sample["api_layer_ms"]
    finite = all(type(x) in (float, int) and math.isfinite(x) for x in values)
    check = dict(frame=sample["frame"], validation_error=error,
                 returned_layer_count=sample["returned_layer_count"],
                 api_sum_ms=sum(values) if finite else None,
                 api_last_ms=values[-1] if values and finite else None,
                 api_calc_time_ms=sample["api_calc_time_ms"], collection_ns=sample["collection_ns"],
                 read_tile_counts=sample["read_tile_counts"], write_tile_counts=sample["write_tile_counts"])
    return check, durations


def verify_timer_encoding(sample, clock_hz, expected):
    """核对本轮 ARM64 SDK 实现：uint32 累计计数，以 256 Icore 周期为一单位，再相邻作差。

    单位来自已安装 libicraft_xrt 的公开函数实现，原始 MemChunk 再独立复算。
    此核对验证软件换算及存储对应，不独立证明振荡器频率，也不区分算术与 DDR 停顿。
    """
    storages = [value for value in sample.get("timer_storage", []) if value.get("captured")]
    require(len(storages) == 1, "Expected one captured native timer allocation")
    storage = storages[0]
    require(storage["encoding"] == "hex_in_increasing_byte_offset", "Unknown raw timer encoding")
    raw = bytes.fromhex(storage["bytes_hex"])
    require(len(raw) == storage["allocation_bytes"] and len(raw) >= 4 * expected, "Incomplete raw timer storage")
    require(type(clock_hz) in (int, float) and math.isfinite(clock_hz) and clock_hz > 0, "Invalid SDK clock value")
    counters = struct.unpack_from("<{}I".format(expected), raw)
    # SDK 先把 uint64 频率转成 FP32，然后用 FP64 作比例换算；保持该次序。
    converted_hz = struct.unpack("<f", struct.pack("<f", clock_hz))[0]
    microseconds_per_tick = 256000000.0 / converted_hz
    previous, rebuilt = 0.0, []
    for count in counters:
        current = count * microseconds_per_tick
        require(current >= previous, "Raw cumulative timer goes backwards; no wrap assumed for a short inference")
        rebuilt.append((current - previous) / 1000.0)
        previous = current
    actual = layer_durations(sample, expected, "durations")
    error = max(abs(a - b) for a, b in zip(actual, rebuilt))
    require(error <= 1e-9, "API timing differs from raw uint32 storage conversion")
    hardware_ms = sample["api_calc_time_ms"]
    require(type(hardware_ms) in (float, int) and math.isfinite(hardware_ms) and hardware_ms >= 0,
            "Invalid native aggregate timer")
    return dict(raw_api_max_error_ms=error, counter_first=counters[0], counter_last=counters[-1],
                raw_counter_monotonic=True, tick_cycles=256, sdk_clock_hz=converted_hz,
                timer_sum_ms=sum(actual), aggregate_minus_layers_ms=hardware_ms - sum(actual),
                independently_measured_clock=False)


def summarize_placement(deployment):
    rows = []
    for backend in deployment["backends"]:
        values = backend["values"]
        require(len({v["value_id"] for v in values}) == len(values), "Duplicate deployment value identities")
        ocm = [v for v in values if v["is_ocm"] and not v["is_host"]]
        rows.append(dict(backend_id=backend["backend_id"], selected=backend["selected_ocm_strategy"],
                         merge_hardops=backend["merge_hardops"], etm_reuse=backend["etm_reuse"],
                         values=len(values), ocm_values=len(ocm),
                         ocm_real_values=sum(v["real"] for v in ocm),
                         ocm_alias_values=sum(not v["real"] for v in ocm),
                         ocm_value_ids=sorted(v["value_id"] for v in ocm),
                         segments=backend["segments"]))
    require(bool(rows), "No deployed NPU allocation metadata")
    return rows


def analyze_run(root, graph, graph_hash, warmup=3, semantics="unverified"):
    root = Path(root)
    manifest = read_json(root / "contract.json")
    permit = read_json(root / "startup-permit.json")
    outcome = read_json(root / "outcome.json")
    closed = read_json(root / "native-complete.json")
    initialized = read_json(root / "native-initialized.json")
    deployment = read_json(root / "npu-deployment.json")
    preflight = read_json(root / "preflight.json")
    require(manifest["source"] == "file_replay" and manifest["files"]["model.json"] == graph_hash,
            "Run does not use the selected generated graph")
    require(outcome.get("native_exit") == 0 and outcome.get("native_shutdown_complete") is True and
            outcome.get("allocation_count_after") == 0 and outcome.get("boot_id") == permit["boot_id"],
            "Run lacks successful native shutdown proof")
    require(closed["completed"] == manifest["iterations"] and not closed["interrupted"], "Incomplete finite run")
    require(preflight["boot_id"] == closed["boot_id"] == permit["boot_id"] and
            preflight["boot_file"]["sha256"] == manifest["boot_sha256"] and preflight["libraries"],
            "Actual BOOT or SDK evidence is missing")
    strategy, samples = manifest.get("ocm_strategy", -1), manifest.get("diagnostic_samples", 0)
    require(permit.get("ocm_strategy", -1) == deployment["requested_ocm_strategy"] == strategy and
            permit.get("diagnostic_samples", 0) == samples, "OCM strategy or sampling permit drifted")
    require(0 <= warmup < manifest["iterations"] - samples and samples > 0, "No ordinary steady samples")
    allocation = summarize_placement(deployment)
    if strategy >= 0:
        require(all(b["selected"] == strategy for b in allocation), "Requested OCM policy was not applied")
    mapping = map_layers(graph, initialized)
    frames = [read_json(root / ("native-frame-{}.json".format(i))) for i in range(manifest["iterations"])]
    epoch = frames[0]["epoch"]
    for i, frame in enumerate(frames):
        require(frame["frame"] == i and frame["epoch"] == epoch and frame["same_as_first_raw"] is True and
                frame["host_compute_callbacks"] == 0 and frame["completed_layers"] == initialized["completion_layers"],
                "Frame identity, complete output or native completion mismatch")
        require(frame["timer_sample_collected"] is (i >= len(frames) - samples), "Diagnostic sampling differs")
    steady = frames[warmup:len(frames) - samples]
    timing = {key.replace("_ns", "_ms"): statistics_ms([f["times"][key] / 1e6 for f in steady])
              for key in frames[0]["times"]}
    timing["forward_wait_ms"] = statistics_ms([(f["times"]["forward_ns"] + f["times"]["final_wait_ns"]) / 1e6
                                               for f in steady])
    timing["complete_call_ms"] = statistics_ms([(f["completed_ns"] - f["captured_ns"]) / 1e6 for f in steady])
    timer_checks, observed = [], []
    for frame in frames[-samples:]:
        sample = read_json(root / ("npu-timers-{}.json".format(frame["frame"])))
        require((sample["epoch"], sample["frame"]) == (epoch, frame["frame"]), "Timer belongs to another frame")
        check, durations = inspect_timer(sample, initialized["completion_layers"], semantics)
        if sample.get("timer_storage"):
            try:
                require(preflight["libraries"].get("libicraft_xrt.so") == TIMER_XRT_SHA256,
                        "Raw timer encoding has not been inspected for this SDK build")
                check["encoding_check"] = verify_timer_encoding(sample, deployment["sdk_icore_hz"],
                                                                 initialized["completion_layers"])
            except (ValueError, KeyError, OverflowError, struct.error) as problem:
                check["encoding_error"] = str(problem)
                if semantics != "unverified":
                    raise
        timer_checks.append(check)
        if durations is not None:
            observed.append(durations)
    ranked, kinds = [], defaultdict(float)
    if observed:
        for op in mapping:
            start, count = op["first_layer"], op["layer_count"]
            measured = [sum(v[start:start + count]) for v in observed]
            row = dict(op, milliseconds=statistics_ms(measured))
            kinds[op["kind"]] += row["milliseconds"]["mean"]
            ranked.append(row)
        ranked.sort(key=lambda value: value["milliseconds"]["mean"], reverse=True)
    raw = root / "native-first-output.f32"
    # 历史记录没有尺寸字段，仍对应固定 640×640 的 one-to-one 图。
    width, height = manifest.get("input_size", (640, 640))
    anchors = sum((width // stride) * (height // stride) for stride in (8, 16, 32))
    require(raw.stat().st_size == 84 * anchors * 4, "Complete dense output is missing")
    floats = array("f")
    floats.frombytes(raw.read_bytes())
    if sys.byteorder != "little":
        floats.byteswap()
    require(floats.itemsize == 4 and all(math.isfinite(x) for x in floats), "Non-finite model output")
    return dict(directory=str(root.resolve()), boot_id=permit["boot_id"], strategy=strategy,
                identities={key: manifest["files"][key] for key in ("model.json", "model.raw", "input.rgb", "yolo26_infer_probe")},
                boot_sha256=manifest["boot_sha256"], sdk_libraries=preflight["libraries"],
                ordinary_frame_range=[warmup, len(frames) - samples],
                timings=timing, placement=allocation, timer_semantics=semantics,
                sdk_clock_hz=dict(icore=deployment["sdk_icore_hz"], ddr=deployment["sdk_ddr_hz"]),
                clock_values_are_measured=deployment["clock_values_are_measured"], timer_checks=timer_checks,
                kinds_ms=dict(sorted(kinds.items(), key=lambda pair: pair[1], reverse=True)),
                ranked_operations=ranked, output_sha256=digest(raw), output_finite=True)


def compare_runs(reports):
    require(bool(reports), "No runs to compare")
    reference = reports[0]
    for run in reports:
        require(run["identities"] == reference["identities"] and run["boot_sha256"] == reference["boot_sha256"] and
                run["sdk_libraries"] == reference["sdk_libraries"],
                "Model, input, executable, SDK or BOOT differs between OCM runs")
    require(len({run["boot_id"] for run in reports}) == len(reports), "OCM batches reused a consumed startup")
    require(len({run["strategy"] for run in reports}) == len(reports), "Duplicate OCM comparison strategy")
    return dict(reference_strategy=reference["strategy"],
                all_complete_outputs_bitwise_equal=all(r["output_sha256"] == reference["output_sha256"] for r in reports),
                rows=[dict(strategy=r["strategy"], selected=[b["selected"] for b in r["placement"]],
                           forward_wait_ms=r["timings"]["forward_wait_ms"],
                           complete_call_ms=r["timings"]["complete_call_ms"], kinds_ms=r["kinds_ms"]) for r in reports])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timer-semantics", choices=("unverified", "durations", "cumulative"), default="unverified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph, identity = read_json(args.graph), digest(args.graph)
    reports = [analyze_run(root, graph, identity, args.warmup, args.timer_semantics) for root in args.run]
    result = dict(schema=1, comparison=compare_runs(reports), runs=reports,
                  timer_semantics_explicitly_selected=args.timer_semantics != "unverified",
                  board_access=False, new_inference_executed=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
