"""核验原生计数器时间线；不把 Tile 计数或层进度转换成 DDR 字节及 MPE 停顿。"""
import argparse
import json
from pathlib import Path

from common import save_json, sha256
from probe_contract import require


def analyze(trace, expected_layers):
    require(trace.get("schema") == 1 and trace.get("clock") == "host_steady_clock_ns",
            "Unknown counter timeline format")
    columns, rows = trace["column_ids"], trace["samples"]
    require(type(expected_layers) is int and 0 < expected_layers <= 10000, "Invalid completed-layer bound")
    require(columns == list(range(1, len(columns) + 1)) and 1 <= len(columns) <= 4,
            "Expected explicit physical columns 1..cols")
    require(2 <= len(rows) <= 512 and type(trace["truncated"]) is bool, "Invalid bounded trace length")
    require(trace.get("register_reads_are_atomic") is False and trace.get("ddr_bytes_measured") is False and
            trace.get("mpe_stall_cycles_measured") is False, "Unexpected measurement semantics")
    prior_end, prior_layer = 0, 0
    for row in rows:
        begin, end = row["begin_ns"], row["end_ns"]
        before, after = row["layer_before"], row["layer_after"]
        require(all(type(v) is int for v in (begin, end, before, after)) and
                0 < begin <= end and begin >= prior_end, "Overlapping or reversed observation interval")
        require(prior_layer <= before <= after <= expected_layers, "Nonmonotonic or invalid layer progress")
        for field in ("reads", "writes"):
            require(len(row[field]) == len(columns) and
                    all(type(v) is int and 0 <= v < 2 ** 32 for v in row[field]), "Invalid uint32 counter vector")
        prior_end, prior_layer = end, after
    require(rows[-1]["layer_after"] == expected_layers, "Final observation does not prove expected layer count")
    counters = []
    for index, column in enumerate(columns):
        for field in ("reads", "writes"):
            values = [row[field][index] for row in rows]
            counters.append(dict(column=column, direction=field, first=values[0], last=values[-1],
                                 changed=any(a != b for a, b in zip(values, values[1:])),
                                 decreases=sum(b < a for a, b in zip(values, values[1:])),
                                 all_zero=not any(values)))
    # 只记录“计数变化落在此区间”，不能把区间长度称为访存耗时或计算等待时间。
    changes = []
    for left, right in zip(rows, rows[1:]):
        if left["layer_after"] != right["layer_after"] or any(left[f] != right[f] for f in ("reads", "writes")):
            changes.append(dict(begin_ns=left["begin_ns"], end_ns=right["end_ns"],
                                layer_from=left["layer_after"], layer_to=right["layer_after"],
                                read_delta_mod_uint32=[(b - a) % 2 ** 32 for a, b in zip(left["reads"], right["reads"])],
                                write_delta_mod_uint32=[(b - a) % 2 ** 32 for a, b in zip(left["writes"], right["writes"])]))
    return dict(schema=1, samples=len(rows), observation_span_ms=(rows[-1]["end_ns"] - rows[0]["begin_ns"]) / 1e6,
                register_read_total_ms=sum(row["end_ns"] - row["begin_ns"] for row in rows) / 1e6,
                maximum_sample_gap_ms=max(b["begin_ns"] - a["end_ns"] for a, b in zip(rows, rows[1:])) / 1e6,
                truncated=trace["truncated"], counters=counters, changes=changes,
                tile_activity_observed=any(c["changed"] for c in counters),
                ddr_bandwidth_bytes_per_second=None, mpe_wait_ms=None,
                note="Hardware register observations; Tile units, wrap multiplicity and stall causes are unverified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous observations")
    record = json.loads(args.timers.read_text(encoding="utf-8"))
    result = analyze(record["counter_timeline"], record["expected_layers"])
    result.update(source_sha256=sha256(args.timers), epoch=record["epoch"], frame=record["frame"])
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
