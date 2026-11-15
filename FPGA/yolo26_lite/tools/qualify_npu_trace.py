"""核对访存观察候选的完整实现报告；只沿用原 AI／GT 限定例外，不接触设备。"""
import argparse
import csv
import json
from pathlib import Path
import re

from common import save_json, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timing_summary(text):
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if "WNS(ns)" in line and "TNS(ns)" in line)
    fields = next(line.split() for line in lines[header + 1:] if re.match(r"^\s*-?\d+\.\d+\s", line))
    require(len(fields) == 12, "Inspect changed native timing summary format")
    values = [float(value) for value in fields]
    require(values[0] >= 0 and values[4] >= 0 and all(values[i] == 0 for i in (1, 2, 5, 6)),
            "Complete routed setup/hold failed")
    for pattern in (r"There are\s+0\s+pins that are not constrained for maximum delay\.",
                    r"There are\s+0\s+register/latch pins with no clock\."):
        require(re.search(pattern, text), "Unconstrained internal endpoints or missing register clocks")
    prefix = "U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/"
    allowed = {prefix + "clkout1_buf/I": (1.409, 1.0, -0.409),
               prefix + "plle2_adv_inst/CLKOUT0": (1.071, 1.0, -0.071)}
    failures = []
    for line in lines:
        if not re.match(r"^(Min Period|Max Period|Low Pulse Width|High Pulse Width)\s", line):
            continue
        match = re.search(r"\s(-\d+\.\d+)\s+\S+\s+(\S+)\s*$", line)
        if match:
            actual = tuple(float(x) for x in line.split() if re.fullmatch(r"-?\d+\.\d+", x))
            require(line.startswith("Min Period") and allowed.get(match[2]) == actual,
                    "New or changed pulse-width exception: " + line)
            failures.append(line)
    require(len(failures) == 2 and values[10] == 2, "Expected exactly the two qualified AI exceptions")
    return dict(WNS_ns=values[0], WHS_ns=values[4], TNS_ns=values[1], THS_ns=values[5],
                raw_timing_pass=False, approved_AI_period_exceptions=failures)


def drc_errors(text):
    allowed = {"REQP-52": "rx_sdi_inst/SDI0/SDI/GTX_i/gtxe2_i",
               "REQP-56": "rx_sdi_inst/gtxe2_common_0/gtxe2_common_i"}
    errors = re.findall(r"^(\S+#\d+)\s+Error\s*\n(.*?)(?=^\S+#\d+\s|\Z)", text, re.M | re.S)
    for name, body in errors:
        rule = name.split("#")[0]
        require(rule in allowed and re.findall(r"\bcell\s+(\S+?):", body) == [allowed.get(rule)],
                "New DRC error or changed legacy GT endpoint: " + name)
    # 检查汇总中的规则种类，避免原生详细段格式变化时误认没有错误。
    summary_rules = re.findall(r"^\|\s*(\S+)\s*\|\s*Error\s*\|", text, re.M)
    require(set(summary_rules) == {name.split("#")[0] for name, _ in errors},
            "DRC summary and detail coverage differ")
    return [name for name, _ in errors]


def resources(path):
    labels = {"Slice LUTs": "LUT", "Slice Registers": "FF", "Block RAM Tile": "BRAM36",
              "DSPs": "DSP", "Slice": "Slice"}
    result = {}
    for line in path.read_text().splitlines():
        fields = [field.strip().rstrip("*").strip() for field in line.split("|")]
        if len(fields) > 2 and fields[1] in labels:
            result[labels[fields[1]]] = float(fields[2].replace(",", ""))
    require(set(result) == set(labels.values()), "Native resource report is incomplete: " + str(path))
    return result


def closed_stages(work):
    stages = {}
    for name in ("synthesis-banked", "opt-banked", "place-csi-incremental", "route-csi-incremental",
                 "report-closure"):
        row = json.loads((work / "logs" / (name + ".json")).read_text())
        require(row.get("state") == "completed" and row.get("exit_code") == 0, "Native stage failed: " + name)
        stages[name] = {key: row[key] for key in ("exit_code", "seconds")}
    return stages


def fully_routed_count(route):
    def route_count(label):
        match = re.search(r"# of " + re.escape(label) + r"\.+\s*:\s*(\d+)\s*:", route)
        require(match, "Missing native route count: " + label)
        return int(match[1])

    count = route_count("routable nets")
    require(count > 0 and route_count("fully routed nets") == count and route_count("nets with routing errors") == 0,
            "Routing is incomplete")
    return count


def verify_frozen_sources(work, manifest):
    # 准备器保留原始依赖与覆盖来源，同一个输入以最后一次明确覆盖为准。
    inputs = {row["input"]: row for row in manifest["sources"]}
    for relative, row in inputs.items():
        path = (work / relative).resolve(strict=True)
        require(work in path.parents, "Source manifest escapes build directory")
        require(sha256(path) == row["sha256"], "Frozen implementation input changed: " + relative)


def qualify(work):
    reports = work / "reports/physical-final-closure"
    stages = closed_stages(work)
    timing = timing_summary((reports / "timing-final.rpt").read_text())
    require("VIOLATED" not in (reports / "bus-skew-final.rpt").read_text(), "Bus skew violation")
    count = fully_routed_count((reports / "route-status-final.rpt").read_text())
    legacy_errors = drc_errors((reports / "drc-final.rpt").read_text())
    facts = [line.split("\t") for line in (reports / "final-native-facts.tsv").read_text().splitlines()]
    require(["PART", "xc7z030ffg676-2"] in facts, "Wrong upstream part for the qualified JFM flow")
    require(["DEBUG_CORES", "0"] in facts and ["MARK_DEBUG", "0"] in facts and
            ["FORBIDDEN_MODULES", "0"] in facts, "Unexpected diagnostic core or old instrumentation")
    removed = [row for row in facts if row[0] == "REMOVED"]
    require(len(removed) == 5 and all(row[-1] == "0" for row in removed), "Old diagnostic hierarchy remains")
    retained = [row for row in facts if row[0] == "RETAINED"]
    require({row[1] for row in retained} == {"deferred_control", "startup_drain"} and
            all(int(row[-1]) > 0 for row in retained), "Existing functional consumers are missing")
    for clock, period in (("clk_fpga_0", 20), ("clk_out1_clk_wiz_0", 10), ("clk_pll_i", 5)):
        rows = [row for row in facts if row[:2] == ["CLOCK", clock]]
        require(len(rows) == 1 and float(rows[0][2]) == period, "Functional clock changed: " + clock)
    for group in ("READY", "CONSUMER_RESET"):
        rows = [row for row in facts if row[0] == group]
        require(rows and all(float(row[-2]) == 5 and float(row[-1]) >= 0 for row in rows),
                "Existing readiness/reset paths failed: " + group)
    connected = [row for row in facts if row[0] == "S01_CONNECTED"]
    require({row[1] for row in connected} == {"ARVALID", "ARADDR", "RREADY"} and
            all(row[2] and row[2] != "GND" for row in connected), "HDMI read port disconnected")
    for group in ("CSI_TIMING", "FILTER_TIMING"):
        rows = [row for row in facts if row[0] == group]
        require({row[1] for row in rows} == {"max", "min"} and all(float(row[-1]) >= 0 for row in rows),
                "Existing function timing failed: " + group)
    control = [line.split("\t") for line in (reports / "control-cdc.tsv").read_text().splitlines()]
    require({row[0] for row in control if not row[0].endswith("_ENDPOINT")} == {"TOGGLE", "REPLY", "COMMAND"}
            and all(float(row[-1]) >= 0 for row in control), "Existing mailbox timing failed")
    with (reports / "npu-trace-cdc-endpoints.tsv").open(newline="") as stream:
        cdc = list(csv.DictReader(stream, delimiter="\t"))
    require({row["group"] for row in cdc} == {"request", "response", "command", "result"},
            "Trace mailbox coverage missing")
    for group, period in (("request", 5), ("response", 10), ("command", 5), ("result", 10)):
        rows = [row for row in cdc if row["group"] == group]
        require(len({row["endpoint"] for row in rows}) == len(rows) and
                all(float(row["requirement_ns"]) == period and float(row["slack_ns"]) >= 0 for row in rows),
                "Trace mailbox timing or endpoint coverage failed: " + group)
    for direction in ("from", "to"):
        for mode in ("max", "min"):
            text = (reports / ("npu-trace-" + direction + "-" + mode + ".rpt")).read_text()
            require("Slack (MET)" in text and "VIOLATED" not in text, "Trace local timing failed")
    manifest = json.loads((work / "current-source-manifest.json").read_text())
    verify_frozen_sources(work, manifest)
    total = resources(reports / "total-utilization.rpt")
    trace = resources(reports / "npu-trace-utilization.rpt")
    require(trace["BRAM36"] == 4 and total["BRAM36"] <= 265, "Unexpected trace/total BRAM organization")
    return dict(schema=1, bit_generation_prerequisite_pass=True, **timing, stages=stages,
                routable_nets=count, fully_routed_nets=count, unconstrained_internal_endpoints=0,
                existing_GT_DRC_errors=legacy_errors, new_timing_waivers=False, new_DRC_waivers=False,
                new_false_paths=False, trace_mailbox_endpoints=len(cdc), trace_resources=trace,
                total_resources=total, checkpoint=str(work / "pipeline-routed-csi-incremental.dcp"),
                source_manifest_sha256=sha256(work / "current-source-manifest.json"),
                hardware_profile=manifest["hardware_profile"], csr_base=manifest["csr_base"],
                capture_bytes=manifest["capture_bytes"], board_tested=False,
                CDC_report=str(reports / "npu-trace-cdc.rpt"),
                CDC_scope="held-payload mailbox bounded in both directions; inspect reset synchronizers in native report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    args = parser.parse_args()
    work = args.build.resolve(strict=True)
    result = qualify(work)
    target = work / "reports/qualification-npu-trace.json"
    require(not target.exists(), "Preserve prior qualification result")
    save_json(target, result)
    print(json.dumps(result, indent=2))
