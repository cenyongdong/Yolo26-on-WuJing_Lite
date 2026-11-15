"""从原厂 TaskGen 检查相邻卷积的 KLM 容量与覆盖关系；不改指令、不把容量证明当成预取完成。"""
import argparse
from collections import defaultdict
from pathlib import Path
import re

from common import save_json, sha256
from probe_contract import require


TASK = re.compile(r"^\[(\d+)\]%(\d+) .* : (\w+)\(\)")
KLM = re.compile(r"^\s+(\w+)\s+= TILE\.KLM\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]"
                 r"\.(\w+)\.U\[(\d+)\]\.B\[\s*(\d+)\]\.DMA\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]")


def parse_tasks(lines):
    active, complete, current = False, False, None
    accesses = []
    for line in lines:
        if "TaskGen Begin" in line:
            require(not active and not complete, "Repeated TaskGen section")
            active = True
        elif "TaskGen End" in line or "TaskGen Done" in line:
            require(active, "Unmatched TaskGen end")
            active, complete = False, True
            break
        elif active:
            task = TASK.match(line)
            if task:
                current = (int(task[1]), int(task[2]), task[3])
            if "TILE.KLM" not in line:
                continue
            tile = KLM.match(line)
            require(current is not None and tile is not None, "Unknown KLM task representation")
            role, core, column, row, bank, dtype, unit, base, width, height, step_x, step_y = tile.groups()
            core, column, row, bank, unit, base, width, height, step_x, step_y = map(
                int, (core, column, row, bank, unit, base, width, height, step_x, step_y))
            require(dtype == "INT8" and unit == 32 and width > 0 and height > 0 and step_x > 0,
                    "This proof only covers the observed INT8 U32 KLM geometry")
            require((role == "klm" and current[2] == "MatMulTask") or
                    (role == "dst" and current[2] in ("AssignTask", "BoardcastTask")),
                    "Unknown KLM reader or writer")
            # 这是包含步长空洞的保守地址包络；互不相交足以证明无覆盖，不相交失败不证明一定冲突。
            end = base + (height - 1) * step_y + (width - 1) * step_x + 1
            accesses.append(dict(operation=current[0], task=current[1], task_type=current[2],
                                 mpe=(core, column, row, bank), write=role == "dst",
                                 begin=base * unit, end=end * unit,
                                 payload_bytes=width * height * unit))
    require(complete and accesses and len({v["operation"] for v in accesses}) == 1,
            "Incomplete or mixed-operation task log")
    return accesses


def plan(previous, following, capacity_bytes=65536):
    require(capacity_bytes > 0 and capacity_bytes % 32 == 0, "Capacity must be in complete INT8 U32 words")
    old, new = defaultdict(list), defaultdict(list)
    for target, accesses in ((old, previous), (new, following)):
        for access in accesses:
            require(0 <= access["begin"] < access["end"] <= capacity_bytes, "Original KLM access exceeds bound")
            target[access["mpe"]].append(access)
    require(set(new).issubset(old), "Next layer introduces an unexamined MPE")
    shift = max(v["end"] for v in previous)
    shift = (shift + 31) // 32 * 32
    rows, broadcasts = [], {}
    for mpe in sorted(new):
        writers = [v for v in new[mpe] if v["write"]]
        readers = [v for v in new[mpe] if not v["write"]]
        require(writers and readers, "A target MPE lacks a weight producer or consumer")
        first = min(writers, key=lambda v: v["task"])
        prior_readers = [v for v in old[mpe] if not v["write"]]
        require(prior_readers and first["task"] < min(v["task"] for v in readers),
                "Task order does not establish an initial weight load")
        end = max(v["end"] for v in new[mpe]) + shift
        overlap = any(first["begin"] < v["end"] and v["begin"] < first["end"] for v in old[mpe])
        relocated_overlap = any(first["begin"] + shift < v["end"] and
                               v["begin"] < first["end"] + shift for v in old[mpe])
        rows.append(dict(mpe=list(mpe), previous_end=max(v["end"] for v in old[mpe]),
                         previous_last_reader=max(v["task"] for v in prior_readers),
                         first_load_task=first["task"], initial_prefetch_overlaps_previous=overlap,
                         relocated_first_begin=first["begin"] + shift,
                         relocated_first_end=first["end"] + shift, relocated_max_end=end,
                         capacity_fits=end <= capacity_bytes, relocated_overlap=relocated_overlap))
        require(first["task_type"] == "BoardcastTask", "First load is not the observed broadcast form")
        if first["task"] in broadcasts:
            require(broadcasts[first["task"]] == first["payload_bytes"], "Broadcast destinations disagree")
        broadcasts[first["task"]] = first["payload_bytes"]
    return dict(schema=1, capacity_bound_bytes=capacity_bytes, relocation_bytes=shift,
                relocation_u32_words=shift // 32, mpes=rows, first_broadcast_tasks=sorted(broadcasts),
                first_broadcast_logical_bytes=sum(broadcasts.values()),
                capacity_and_nonoverlap=all(v["capacity_fits"] and not v["relocated_overlap"] for v in rows),
                previous_address_union_conservatively_reserved=True,
                task_logs_are_not_full_precision_serialization=True,
                native_dependency_regeneration_verified=False, hardware_port_overlap_verified=False,
                full_numerical_prefetch_verified=False, deployable=False,
                required_changes=["Relocate every next-layer KLM reader and writer consistently",
                                  "Bind next-layer weights to the earlier HardOp without changing parameter bytes",
                                  "Regenerate cross-layer completion/state dependencies from exact native tasks",
                                  "Remove redundant initial loads only after dependency and full-output simulation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--following", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous address proof")
    accesses = []
    for path in (args.previous, args.following):
        with path.open(encoding="utf-8") as stream:
            accesses.append(parse_tasks(stream))
    result = plan(*accesses)
    result.update(previous_log_sha256=sha256(args.previous), following_log_sha256=sha256(args.following))
    save_json(args.output, result)
    print("KLM capacity/nonoverlap:", result["capacity_and_nonoverlap"], "; deployable:", result["deployable"])


if __name__ == "__main__":
    main()
