"""核验 512×384 视频候选的完整物理报告，供原厂 BIT 和 BOOT 打包使用。"""
import argparse
import csv
import json
from pathlib import Path
import re

from common import save_json, sha256
from qualify_npu_trace import (closed_stages, drc_errors, fully_routed_count, require, resources,
                               timing_summary, verify_frozen_sources)


VIDEO_PROFILE = "probe-free-drain-overlay-csi-v3-yolo26-candidate"


def hdmi_clock_periods(text):
    hierarchy = "U0_ps_ai_peripheral/hdmi_display/clock_generator/"
    periods = {}
    for name, pin, mhz in (("pixel", "pixel_buffer/O", 74.25), ("serial", "mmcm/CLKOUT0", 371.25)):
        pattern = r"^\S+\s+(\d+\.\d+)\s+\{[^\n]*\}\s+\S+\s+\{" + re.escape(hierarchy + pin) + r"\}\s*$"
        rows = re.findall(pattern, text, re.M)
        require(len(rows) == 1 and abs(float(rows[0]) - 1000 / mhz) < 0.001,
                "Missing or incorrect 720p HDMI " + name + " clock")
        periods[name] = float(rows[0])
    return periods


def cdc_classifications(path):
    text = path.read_text()
    require("CDC Report" in text, "Missing CDC report: " + path.name)
    rows = re.findall(r"^(CDC-\d+)\s+(\w+)\s+(\d+)\s+", text, re.M)
    require(rows, "Missing native CDC classification table: " + path.name)
    return {name: dict(severity=severity, count=int(count)) for name, severity, count in rows}


def held_dsp_paths(path, expected):
    if not expected:
        return dict(endpoints=0)
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    properties = dict(row for row in rows if len(row) == 2)
    require(properties.get("BREG") == "2" and properties.get("CLK") == "clk_pll_i" and
            "/mailbox/request_sync_reg[2]/C" in properties.get("CEB1", "") and properties.get("CEB2"),
            "Missing DSP receive registers or synchronized enable")
    paths = [row for row in rows if row[0] == "PATH"]
    require(len(paths) == expected and len({row[2] for row in paths}) == expected and
            all(float(row[3]) == 5 and float(row[4]) >= 0 for row in paths),
            "Incomplete or failed held-payload DSP path timing")
    return dict(endpoints=expected, minimum_slack_ns=min(float(row[4]) for row in paths))


def qualify(work):
    manifest_path = work / "current-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["hardware_profile"] == VIDEO_PROFILE and manifest["preprocess_output"] == [512, 384] and
            manifest["display"] == "1280x720@60" and manifest["csr_base"] == "0x400b5000",
            "Unexpected video hardware profile")
    stages = closed_stages(work)
    verify_frozen_sources(work, manifest)
    reports = work / "reports/physical-final-closure"
    timing = timing_summary((reports / "timing-final.rpt").read_text())
    count = fully_routed_count((reports / "route-status-final.rpt").read_text())
    require("VIOLATED" not in (reports / "bus-skew-final.rpt").read_text(), "Bus skew violation")
    legacy_errors = drc_errors((reports / "drc-final.rpt").read_text())
    periods = hdmi_clock_periods((reports / "clocks.rpt").read_text())
    facts = [line.split("\t") for line in (reports / "final-native-facts.tsv").read_text().splitlines()]
    require(["PART", "xc7z030ffg676-2"] in facts and ["DEBUG_CORES", "0"] in facts and
            ["MARK_DEBUG", "0"] in facts and ["FORBIDDEN_MODULES", "0"] in facts, "Routed design identity differs")
    for group in ("CSI_TIMING", "FILTER_TIMING"):
        rows = [row for row in facts if row[0] == group]
        require({row[1] for row in rows} == {"max", "min"} and all(float(row[-1]) >= 0 for row in rows),
                "Unclosed legacy consumer timing: " + group)
    with (reports / "yolo26-cdc-endpoints.tsv").open(newline="") as stream:
        cdc = list(csv.DictReader(stream, delimiter="\t"))
    budgets = {"request": 5, "response": 10, "command": 5, "result": 10, "stop": 5, "link": 10}
    require({row["group"] for row in cdc} == set(budgets), "Missing video CDC group")
    for group, budget in budgets.items():
        rows = [row for row in cdc if row["group"] == group]
        require(len({row["endpoint"] for row in rows}) == len(rows) and
                all(float(row["requirement_ns"]) == budget and float(row["slack_ns"]) >= 0 for row in rows),
                "Video CDC timing or endpoint coverage differs: " + group)
    for direction in ("from", "to"):
        for mode in ("max", "min"):
            text = (reports / ("yolo26-" + direction + "-" + mode + ".rpt")).read_text()
            require("Slack (MET)" in text and "VIOLATED" not in text, "Video local setup/hold failed")
    classifications = {scope: cdc_classifications(reports / (scope + "-cdc.rpt")) for scope in ("yolo26", "hdmi")}
    dsp_cdc = held_dsp_paths(reports / "yolo26-dsp-cdc.tsv",
                             classifications["yolo26"].get("CDC-13", {}).get("count", 0))
    checkpoint = work / "pipeline-routed-csi-incremental.dcp"
    return dict(schema=1, bit_generation_prerequisite_pass=True, **timing, stages=stages,
                routable_nets=count, fully_routed_nets=count, existing_GT_DRC_errors=legacy_errors,
                source_manifest_sha256=sha256(manifest_path), checkpoint_sha256=sha256(checkpoint),
                hardware_profile=VIDEO_PROFILE, preprocess_output=manifest["preprocess_output"],
                display=manifest["display"], csr_base=manifest["csr_base"], hdmi_periods_ns=periods,
                bounded_cdc_endpoints=len(cdc), cdc_classifications=classifications,
                held_dsp_cdc=dsp_cdc,
                yolo26_resources=resources(reports / "yolo26-utilization.rpt"),
                total_resources=resources(reports / "total-utilization.rpt"), board_tested=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    args = parser.parse_args()
    work = args.build.resolve(strict=True)
    result = qualify(work)
    target = work / "reports/qualification-video.json"
    require(not target.exists(), "Preserve prior qualification result")
    save_json(target, result)
    print(json.dumps(result, indent=2))
