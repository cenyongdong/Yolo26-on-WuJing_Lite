"""统计原厂最终指令日志中的 ETM 请求与地址复用；这是静态请求量，不是实测 DDR 总线流量。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from analyze_npu import require
from common import save_json, sha256


MEMORY = re.compile(r"\b(RETM|WETM)\s+\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]\.(\d+)\s+->")


def union_bytes(intervals):
    end, total = -1, 0
    for start, stop in sorted(intervals):
        require(0 <= start < stop, "Invalid memory interval")
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze_text(text):
    """跳过 TaskGen，读取 Instructions Begin/End 之间最终展开的内存指令。

    size0/size1 是向量单位，arsize/awsize 是每单位字节数；step 是单位步长，base 为字节偏移。
    接受正常地址模式的连续／正步长行；逐段计算地址并集，不把跨段空洞算成已搬运数据。
    广播或特殊寻址模式仍拒绝，不猜测其物理事务。
    """
    active = False
    sections = 0
    totals, intervals, operations = {}, defaultdict(list), Counter()
    for line in text.splitlines():
        if "Instructions Begin" in line:
            require(not active and sections == 0, "Repeated instruction section")
            active, sections = True, sections + 1
            continue
        if "Instructions End" in line:
            require(active, "Unmatched instruction end")
            active = False
            continue
        if not active or not line.startswith(">>"):
            continue
        mnemonic = re.search(r"\].\d+\.P\d+ (\w+) ", line)
        if mnemonic:
            operations[mnemonic.group(1)] += 1
        match = MEMORY.search(line)
        if not match:
            # WETM 的状态触发指令不是数据写入。
            if re.search(r"\b(?:RETM|WETM)\b", line):
                require("ZSTS" in line or "ZSTV" in line, "Unknown ETM instruction encoding")
            continue
        direction = match.group(1)
        base, width, height, step_x, step_y, slot = map(int, match.groups()[1:])
        size_key, mode_key = ("arsize", "axmode") if direction == "RETM" else ("awsize", "awmode")
        unit = re.search(r"\b" + size_key + r"=(\d+)", line)
        mode = re.search(r"\b" + mode_key + r"=(\d+)", line)
        require(unit is not None and mode is not None and int(mode.group(1)) == 0,
                "Only decoded normal ETM addressing is supported")
        unit = int(unit.group(1))
        require(unit > 0 and width > 0 and height > 0 and step_x >= 1, "Invalid stride or empty ETM request")
        key = (direction, slot)
        record = totals.setdefault(key, dict(direction=direction, buffer_slot=slot, requests=0, requested_bytes=0))
        record["requests"] += 1
        record["requested_bytes"] += width * height * unit
        for row in range(height):
            start = base + row * step_y * unit
            if step_x == 1:
                intervals[key].append((start, start + width * unit))
            else:
                # 例如转置读取的一个向量可以是一整块 20×20×16，而非一个 AXI beat。
                # arsize/awsize 是日志解码后的向量字节数，按显式步长列举有效段。
                intervals[key].extend((start + column * step_x * unit,
                                       start + (column * step_x + 1) * unit) for column in range(width))
    require(sections == 1 and not active and totals, "Incomplete or missing instruction log")
    rows = []
    for key, record in sorted(totals.items()):
        unique = union_bytes(intervals[key])
        rows.append(dict(record, unique_address_bytes=unique,
                         request_to_unique_ratio=record["requested_bytes"] / unique))
    return dict(memory=rows, instructions=dict(operations),
                requested_read_bytes=sum(r["requested_bytes"] for r in rows if r["direction"] == "RETM"),
                requested_write_bytes=sum(r["requested_bytes"] for r in rows if r["direction"] == "WETM"),
                actual_ddr_bytes_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--operation", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous instruction analysis")
    results = []
    for identity in args.operation:
        path = args.logs / ("hardOp" + str(identity) + "_zg_insts.log")
        result = analyze_text(path.read_text(encoding="utf-8"))
        results.append(dict(result, operation=identity, log_sha256=sha256(path)))
    save_json(args.output, dict(schema=1, operations=results, scope="static_final_instruction_requests"))
    print(json.dumps([dict(operation=r["operation"], memory=r["memory"]) for r in results], indent=2))


if __name__ == "__main__":
    main()
