"""从合格依赖包准备独立 YOLO26 SoC；不运行工具、不连接设备、不修改厂商安装。"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from common import sha256

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "FPGA/yolo26_lite"
sys.path.insert(0, str(ROOT / "tools/release"))
import prepare_refactored_fpga as maintained


RTL = ("dma_descriptor", "dma_mm2s", "color_pipeline", "pixel_unpack", "resize_letterbox",
       "pixel_pack", "dma_s2mm", "stream_register", "preprocess", "register_bank", "mailbox", "command_dispatch",
       "control", "peripheral", "axi_read_share", "axi_write_share", "soc_bridge")
LEGACY_CONSTRAINTS = ("ps_release_factory_clocks.tcl", "hdmi_cdc.tcl", "consumer_capture_scope.tcl")


def validate_build_location(output):
    # 原厂执行保护按安装根目录识别写入，构建不能放在这些目录内。
    normalized = output.resolve().as_posix().lower()
    for protected in ("d:/vivado18.3", "d:/xilinx", "c:/fudanmicro/procise", "c:/icraft"):
        if normalized == protected or normalized.startswith(protected + "/"):
            raise ValueError("Build directory must be outside tool installations: " + str(output))


def insert_before(path, anchor, addition):
    source = path.read_text(encoding="utf-8-sig")
    if source.count(anchor) != 1:
        raise ValueError("Qualified flow anchor missing or ambiguous: " + str(path))
    path.write_text(source.replace(anchor, addition + anchor), encoding="utf-8")


def exclude_new_hierarchy(path):
    """旧例外的旧对象保持不变；新增模块不能继承时钟对范围的 false path。"""
    source = path.read_text(encoding="utf-8-sig")
    negative = "NAME !~ U0_ps_ai_peripheral/presentation_filter/*"
    positive = "NAME =~ U0_ps_ai_peripheral/presentation_filter/*"
    if negative not in source or "U0_ps_ai_peripheral/yolo26_video/*" in source:
        raise ValueError("Unexpected factory exception scope: " + str(path))
    source = source.replace(negative, negative + " && NAME !~ U0_ps_ai_peripheral/yolo26_video/*")
    source = source.replace(positive, positive + " || NAME =~ U0_ps_ai_peripheral/yolo26_video/*")
    path.write_text(source, encoding="utf-8")


def prepare(reference, output, procise, placement_reference=None, placement_reuse="whole"):
    validate_build_location(output)
    if placement_reuse != "whole" and (placement_reuse != "npu-mig" or placement_reference is None):
        raise ValueError("Partial layout reuse requires a routed placement reference")
    if placement_reference is not None:
        stage = json.loads((placement_reference.parent / "logs/route-csi-incremental.json").read_text())
        if placement_reference.suffix != ".dcp" or stage.get("exit_code") != 0 or stage.get("state") != "completed":
            raise ValueError("Incremental reference requires a completed native routed checkpoint")
    maintained.prepare(reference, output, procise)
    manifest_path = output / "current-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if placement_reference is not None:
        target = output / "inputs/FPGA/reference/csi-v2-placement-reference.dcp"
        shutil.copyfile(placement_reference, target)
        manifest["sources"].append(dict(source=str(placement_reference),
                                        input=target.relative_to(output).as_posix(), sha256=sha256(target)))
        manifest["incremental_reference_scope"] = "placement and routing seed only"
        if placement_reuse == "npu-mig":
            place = output / "place-csi-incremental.tcl"
            anchor = "read_checkpoint -incremental "
            insert_before(place, anchor,
                          "set y26_seed [get_cells -quiet {U0_ps_ai_peripheral/ZGolo U0_ps_ai_peripheral/mig_ddr3_u0}]\n"
                          "if {[llength $y26_seed] != 2} {error \"Partial reuse roots missing\"}\n")
            source = place.read_text(encoding="utf-8")
            place.write_text(source.replace(anchor, anchor + "-reuse_objects $y26_seed "), encoding="utf-8")
            manifest["sources"].append(dict(source="generated:yolo26_scoped_reuse", input=place.name,
                                            sha256=sha256(place)))
            manifest["incremental_reference_scope"] = (
                "reuse only original ZGolo NPU and MIG DDR hierarchy; new placement unfixed")
    copied_rtl = []
    relatives = ["rtl/y26_" + name + ".sv" for name in RTL]
    relatives += ["constraints/y26_" + name + ".tcl" for name in ("control_cdc", "soc_cdc", "soc_validate")]
    for relative in relatives:
        source = MODULE / relative
        target = output / "inputs/FPGA/yolo26_lite" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["sources"].append(dict(source=source.relative_to(ROOT).as_posix(),
                                        input=target.relative_to(output).as_posix(), sha256=sha256(target)))
        if target.suffix == ".sv":
            copied_rtl.append(target)

    synthesis = output / "synthesis-banked.tcl"
    insert_before(synthesis, "read_verilog -library xil_defaultlib -sv {",
                  "set_property verilog_define {YOLO26_CANDIDATE} [current_fileset]\n")
    insert_before(synthesis, "synth_design -top ai7030_edif_top",
                  "read_verilog -sv [list " + " ".join("{" + path.as_posix() + "}" for path in copied_rtl) + "]\n")
    constraints = output / "inputs/FPGA/yolo26_lite/constraints"
    validation = constraints / "y26_soc_validate.tcl"
    insert_before(output / "opt-banked.tcl", "opt_design\n",
                  "set out {" + (output / "reports/yolo26-preopt").as_posix() + "}\nfile mkdir $out\n"
                  "set y26_check_slack 0\nsource {" + validation.as_posix() + "}\nunset y26_check_slack\n")
    legacy = [output / "inputs/FPGA/constraints" / name for name in LEGACY_CONSTRAINTS]
    for path in legacy:
        exclude_new_hierarchy(path)
    closure = """
set y26_cells [get_cells -quiet U0_ps_ai_peripheral/yolo26_video]
if {[llength $y26_cells]!=1} {error "YOLO26 SoC hierarchy missing"}
report_cdc -details -cells $y26_cells -file $out/yolo26-cdc.rpt
report_utilization -cells $y26_cells -file $out/yolo26-utilization.rpt
set y26_registers [get_cells -hier -filter {NAME =~ U0_ps_ai_peripheral/yolo26_video/* && IS_SEQUENTIAL == 1}]
set y26_timing_failed 0
foreach direction {from to} {
    foreach mode {max min} {
        set paths [get_timing_paths -delay_type $mode -$direction $y26_registers -max_paths 10]
        report_timing -delay_type $mode -$direction $y26_registers -max_paths 20 -file $out/yolo26-$direction-$mode.rpt
        if {![llength $paths]} {set y26_timing_failed 1}
        foreach path $paths {
            if {[get_property SLACK $path]<0} {set y26_timing_failed 1}
        }
    }
}
source {VALIDATION}
if {$y26_timing_failed} {error "YOLO26 SoC setup/hold failed; reports preserved"}
""".replace("VALIDATION", validation.as_posix())
    # 先保留旧消费链的查询顺序，再独立检查 YOLO 的实际已布线端点。
    insert_before(output / "report-closure.tcl", "report_utilization -file $out/total-utilization.rpt", closure)
    for path in (synthesis, output / "opt-banked.tcl", output / "report-closure.tcl", *legacy):
        manifest["sources"].append(dict(source="generated:yolo26_soc", input=path.relative_to(output).as_posix(),
                                        sha256=sha256(path)))
    manifest.update(hardware_profile="probe-free-drain-overlay-csi-v3-yolo26-candidate",
                    csr_base="0x400b5000", csr_bytes=256, preprocess_output=[512, 384], display="1280x720@60",
                    gp_clock_ns=10, hp_clock_ns=5, board_tested=False, deployment_qualified=False,
                    timing_constraints="bounded CDC; old exceptions exclude new hierarchy; no new waivers")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared isolated YOLO26 SoC:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--placement-reference", type=Path)
    parser.add_argument("--placement-reuse", choices=("whole", "npu-mig"), default="whole")
    parser.add_argument("--procise", type=Path, default=Path("C:/FudanMicro/Procise"))
    args = parser.parse_args()
    prepare(args.reference_package.resolve(strict=True), args.output.resolve(), args.procise.resolve(strict=True),
            args.placement_reference.resolve(strict=True) if args.placement_reference else None,
            args.placement_reuse)
