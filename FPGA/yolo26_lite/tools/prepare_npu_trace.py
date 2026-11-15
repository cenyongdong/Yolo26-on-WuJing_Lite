"""从合格依赖包准备独立 NPU 访存观察工程；不启用 YOLO 视频候选、不连接板卡。"""
import argparse
import json
from pathlib import Path
import shutil

from common import sha256
from prepare_soc import ROOT, MODULE, maintained, validate_build_location, insert_before, LEGACY_CONSTRAINTS


def prepare(reference, output, procise, placement_reference=None):
    validate_build_location(output)
    maintained.prepare(reference, output, procise)
    manifest_path = output / "current-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if placement_reference is not None:
        # 仅复用 placement/routing seed；它可以尚未时序闭合，绝不借此取得部署资格。
        # 文件名沿用依赖包入口，实际来源及哈希单独覆盖记录。
        stage = json.loads((placement_reference.parent / "logs/route-csi-incremental.json").read_text())
        if placement_reference.suffix != ".dcp" or stage.get("exit_code") != 0 or stage.get("state") != "completed":
            raise ValueError("Incremental reference requires a completed native routed checkpoint")
        target = output / "inputs/FPGA/reference/csi-v2-placement-reference.dcp"
        shutil.copyfile(placement_reference, target)
        manifest["sources"].append(dict(source=str(placement_reference), input=target.relative_to(output).as_posix(),
                                        sha256=sha256(target)))
        manifest["incremental_reference_scope"] = "placement seed only; no timing or deployment qualification inherited"
    rtl = []
    paths = ["rtl/y26_" + name + ".sv" for name in
             ("npu_bus_bins", "npu_trace_store", "npu_trace_csr", "mailbox")]
    paths.append("constraints/y26_npu_trace_cdc.tcl")
    for relative in paths:
        source = MODULE / relative
        target = output / "inputs/FPGA/yolo26_lite" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["sources"].append(dict(source=source.relative_to(ROOT).as_posix(),
                                        input=target.relative_to(output).as_posix(), sha256=sha256(target)))
        if target.suffix == ".sv":
            rtl.append(target)
    synthesis = output / "synthesis-banked.tcl"
    insert_before(synthesis, "read_verilog -library xil_defaultlib -sv {",
                  "set_property verilog_define {YOLO26_NPU_TRACE} [current_fileset]\n")
    insert_before(synthesis, "synth_design -top ai7030_edif_top",
                  "read_verilog -sv [list " + " ".join("{" + p.as_posix() + "}" for p in rtl) + "]\n")
    legacy = [output / "inputs/FPGA/constraints" / name for name in LEGACY_CONSTRAINTS]
    for path in legacy:
        text = path.read_text(encoding="utf-8-sig")
        old = "NAME !~ U0_ps_ai_peripheral/presentation_filter/*"
        if old not in text:
            raise ValueError("Qualified exception scope is absent: " + str(path))
        text = text.replace(old, old + " && NAME !~ U0_ps_ai_peripheral/npu_trace*")
        old = "NAME =~ U0_ps_ai_peripheral/presentation_filter/*"
        text = text.replace(old, old + " || NAME =~ U0_ps_ai_peripheral/npu_trace*")
        path.write_text(text, encoding="utf-8")
    constraint = output / "inputs/FPGA/yolo26_lite/constraints/y26_npu_trace_cdc.tcl"
    insert_before(output / "opt-banked.tcl", "opt_design\n",
                  "set out {" + (output / "reports/npu-trace-preopt").as_posix() + "}\nfile mkdir $out\n"
                  "set trace_check_slack 0\nsource {" + constraint.as_posix() + "}\nunset trace_check_slack\n")
    closure = """
set trace_check_slack 1
source {CONSTRAINT}
set trace_cells [get_cells -quiet U0_ps_ai_peripheral/npu_trace]
if {[llength $trace_cells]!=1} {error "NPU trace hierarchy missing"}
report_cdc -details -cells $trace_cells -file $out/npu-trace-cdc.rpt
report_utilization -cells $trace_cells -file $out/npu-trace-utilization.rpt
set trace_registers [get_cells -hier -filter {NAME =~ U0_ps_ai_peripheral/npu_trace/* && IS_SEQUENTIAL == 1}]
set trace_timing_failed 0
foreach direction {from to} {
 foreach mode {max min} {
  set paths [get_timing_paths -delay_type $mode -$direction $trace_registers -max_paths 10]
  report_timing -delay_type $mode -$direction $trace_registers -max_paths 20 -file $out/npu-trace-$direction-$mode.rpt
  if {![llength $paths]} {set trace_timing_failed 1}
  foreach path $paths {if {[get_property SLACK $path]<0} {set trace_timing_failed 1}}
 }
}
if {$trace_timing_failed} {error "NPU trace setup/hold failed; reports retained"}
""".replace("CONSTRAINT", constraint.as_posix())
    insert_before(output / "report-closure.tcl", "set f [open $out/final-native-facts.tsv w]", closure)
    for path in (synthesis, output / "opt-banked.tcl", output / "report-closure.tcl", *legacy):
        manifest["sources"].append(dict(source="generated:npu_trace", input=path.relative_to(output).as_posix(),
                                        sha256=sha256(path)))
    manifest.update(hardware_profile="probe-free-drain-overlay-csi-v3-npu-trace",
                    csr_base="0x400b6000", csr_bytes=256, capture_bytes=16384,
                    board_tested=False, deployment_qualified=False,
                    timing_constraints="explicit bounded mailbox CDC; new hierarchy excluded from old exceptions")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared NPU trace implementation:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--procise", type=Path, default=Path("C:/FudanMicro/Procise"))
    parser.add_argument("--placement-reference", type=Path)
    args = parser.parse_args()
    prepare(args.reference_package.resolve(strict=True), args.output.resolve(), args.procise.resolve(strict=True),
            args.placement_reference.resolve(strict=True) if args.placement_reference else None)
