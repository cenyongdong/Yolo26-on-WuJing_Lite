"""核验片上实际接口记录，输出分箱时间线；不把接口静默或背压猜测为 MPE 停顿。"""
import argparse
import json
from pathlib import Path

from common import save_json, sha256
from probe_contract import require


FIELDS = ("ar_accepted", "ar_backpressure", "r_accepted", "r_backpressure",
          "aw_accepted", "aw_backpressure", "accepted_write_byte_enables",
          "b_accepted", "b_backpressure", "no_valid_cycles")


def analyze(trace, counters=None):
    require(trace.get("schema") == 1 and trace.get("clock_period_ns_from_constraints") == 5,
            "Unqualified capture format or clock assumption")
    period, capacity, status = trace["period_cycles"], trace["capacity"], trace["status"]
    require(all(type(value) is int for value in (period, capacity, status)) and
            0 < period <= 1000000 and 0 < capacity <= 512 and 0 <= status < 2 ** 32,
            "Invalid bounded capture configuration")
    require(status & 0xfffb == 2, "Capture is not frozen or has a configuration error")
    records = trace["records"]
    require(isinstance(records, list) and 0 < len(records) <= capacity and status >> 16 == len(records),
            "Capture count differs from committed RAM range")
    times = [trace[name] for name in ("arm_begin_ns", "arm_end_ns", "stop_begin_ns", "stop_end_ns")]
    require(all(type(value) is int and value > 0 for value in times) and times == sorted(times),
            "Invalid host command time bounds")
    total = dict.fromkeys(FIELDS, 0)
    timeline, previous_end, origin = [], None, None
    for index, words in enumerate(records):
        require(isinstance(words, list) and len(words) == 16 and
                all(type(word) is int and 0 <= word < 2 ** 32 for word in words), "Invalid 64-byte record")
        tick = words[2] | words[3] << 32
        cycles = words[1]
        require(words[0] == index and 0 < cycles <= period and
                (previous_end is None or tick == previous_end), "Discontinuous capture interval or index")
        require(index == len(records) - 1 or cycles == period, "Only the final bin may be partial")
        for a, b in ((4, 5), (6, 7), (8, 9), (11, 12)):
            require(words[a] + words[b] <= cycles, "Handshake and stall exceed available cycles")
        require(words[10] <= words[8] * 64 and words[13] <= cycles, "Invalid write byte or silent cycle count")
        require(words[14] < 2 ** 27 and words[15] < 2 ** 27 and
                (words[4] > 0 or words[14:16] == [0, 0]), "Invalid raw read address")
        previous_end = tick + cycles
        origin = tick if origin is None else origin
        values = dict(zip(FIELDS, words[4:14]))
        for name, value in values.items():
            total[name] += value
        begin_offset_ns, end_offset_ns = (tick - origin) * 5, (tick + cycles - origin) * 5
        row = dict(index=index, begin_us=begin_offset_ns / 1000, end_us=end_offset_ns / 1000,
                   cycles=cycles, **values, first_ar_raw=words[14], last_ar_raw=words[15],
                   r_handshake_cycle_fraction=words[6] / cycles,
                   ar_stall_given_valid=words[5] / (words[4] + words[5]) if words[4] + words[5] else None,
                   no_valid_cycle_fraction=words[13] / cycles,
                   begin_host_ns_bounds=[times[0] + begin_offset_ns, times[1] + begin_offset_ns],
                   end_host_ns_bounds=[times[0] + end_offset_ns, times[1] + end_offset_ns])
        if counters is not None:
            samples = counters["samples"]
            # 保守界：起点以前确定已完成的层，到终点以后首次观察可能已完成的层。
            before = [s["layer_before"] for s in samples if s["end_ns"] <= row["begin_host_ns_bounds"][0]]
            after = [s["layer_after"] for s in samples if s["begin_ns"] >= row["end_host_ns_bounds"][1]]
            row["completed_layer_range"] = [max(before) if before else 0, after[0] if after else None]
        timeline.append(row)
    cycles = sum(row["cycles"] for row in timeline)
    duration_ns = cycles * 5
    return dict(schema=1, bins=len(records), cycles=cycles, duration_ms=duration_ns / 1e6,
                clock_basis="existing 5ns implementation constraint; not an independent frequency measurement",
                host_alignment_uncertainty_ns=times[1] - times[0], overflow=bool(status & 4),
                capacity_reached=len(records) == capacity, totals=total, timeline=timeline,
                interface_return_payload_bytes=total["r_accepted"] * 64,
                interface_return_payload_MBps=total["r_accepted"] * 64 * 1000 / duration_ns,
                ddr_bandwidth_bytes_per_second=None, mpe_wait_ms=None,
                note="512-bit interface beat payload includes padding; DDR transactions and internal stalls are unobserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous hardware observations")
    record = json.loads(args.timers.read_text(encoding="utf-8"))
    counters = record.get("counter_timeline")
    if counters is not None:
        from analyze_counter_timeline import analyze as validate_counters
        validate_counters(counters, record["expected_layers"])
    result = analyze(record["bus_timeline"], counters)
    result.update(source_sha256=sha256(args.timers), epoch=record["epoch"], frame=record["frame"])
    save_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "timeline"}, indent=2))


if __name__ == "__main__":
    main()
