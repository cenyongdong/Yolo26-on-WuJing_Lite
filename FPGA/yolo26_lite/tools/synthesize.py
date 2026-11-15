"""使用既有 2018.3/JFM 代理器件做独立综合；不生成 BIT，不认证整机时序。"""
from pathlib import Path
import argparse
import shutil
from common import run_logged, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top", choices=("preprocess", "peripheral"), default="preprocess")
    parser.add_argument("--vivado-bin", type=Path, default=Path("D:/vivado18.3/Vivado/2018.3/bin"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ("y26_dma_descriptor", "y26_dma_mm2s", "y26_color_pipeline", "y26_pixel_unpack", "y26_resize_letterbox", "y26_pixel_pack",
             "y26_dma_s2mm", "y26_stream_register", "y26_preprocess")
    if args.top == "peripheral":
        names += ("y26_register_bank", "y26_mailbox", "y26_command_dispatch", "y26_control", "y26_peripheral")
    sources = [root / "rtl" / (name + ".sv") for name in names]
    for path in sources:
        shutil.copyfile(path, output / path.name)
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    cdc_script = ""
    cdc_reports = ""
    if args.top == "peripheral":
        cdc = root / "constraints" / "y26_control_cdc.tcl"
        shutil.copyfile(cdc, output / cdc.name)
        source_hashes[str(cdc.relative_to(root))] = sha256(cdc)
        cdc_script = "source y26_control_cdc.tcl"
        cdc_reports = "report_cdc -details -file cdc.rpt\nreport_exceptions -coverage -file exception_coverage.rpt"
    save_json(output / "sources.json", source_hashes)
    # HP2 原时钟为 200MHz。这里只设置边界预算，不能代替整机实际 IO/CDC 约束。
    constraints = (
        "create_clock -name estimate_clk -period 5.000 [get_ports clk]\n"
        "set_input_delay 1.000 -clock estimate_clk [get_ports -filter {DIRECTION == IN && NAME != clk}]\n"
        "set_output_delay 1.000 -clock estimate_clk [all_outputs]\n")
    if args.top == "peripheral":
        constraints = """create_clock -name gp_clk -period 10.000 [get_ports gp_clk]
create_clock -name hp_clk -period 5.000 [get_ports hp_clk]
set_input_delay 1.000 -clock gp_clk [get_ports {gp_resetn write_valid write_offset* write_data* write_strobe* read_valid read_offset* read_response_ready}]
set_input_delay 1.000 -clock hp_clk [get_ports -filter {DIRECTION == IN && (NAME == hp_resetn || NAME =~ m_axi_*)}]
set_output_delay 1.000 -clock gp_clk [get_ports {write_ready read_ready read_response_valid read_response_data*}]
set_output_delay 1.000 -clock hp_clk [get_ports -filter {DIRECTION == OUT && NAME =~ m_axi_*}]
"""
    (output / "estimate.xdc").write_text(constraints, encoding="ascii")
    script = """if {[catch {
set_param general.maxThreads 2
create_project -in_memory -part xc7z030ffg676-2
read_verilog -sv [list SOURCE_LIST]
read_xdc estimate.xdc
synth_design -mode out_of_context -top TOP_NAME -part xc7z030ffg676-2
CDC_SCRIPT
report_utilization -file utilization.rpt
report_utilization -hierarchical -file hierarchy.rpt
report_timing_summary -delay_type min_max -check_timing_verbose -file timing_estimate.rpt
report_timing -delay_type max -max_paths 20 -file worst_paths.rpt
CDC_REPORTS
write_checkpoint synthesized.dcp
} result]} {puts stderr $result; exit 1}
exit 0
""".replace("SOURCE_LIST", " ".join("{" + path.name + "}" for path in sources))
    script = script.replace("TOP_NAME", "y26_" + args.top).replace("CDC_SCRIPT", cdc_script)
    script = script.replace("CDC_REPORTS", cdc_reports)
    (output / "synthesize.tcl").write_text(script, encoding="ascii")
    native = run_logged([args.vivado_bin / "vivado.bat", "-mode", "batch", "-notrace",
                         "-source", output / "synthesize.tcl"], output, "synthesize")
    required = ("utilization.rpt", "hierarchy.rpt", "timing_estimate.rpt", "worst_paths.rpt", "synthesized.dcp")
    if args.top == "peripheral":
        required += ("cdc.rpt", "exception_coverage.rpt")
    if not all((output / name).is_file() for name in required):
        raise RuntimeError("Native synthesis missed required artifacts")
    save_json(output / "result.json", dict(native=native, top="y26_" + args.top, proxy_part="xc7z030ffg676-2",
              physical_target="JFMQL30TAI676H", synthesis_passed=True, timing_budget_ns=5,
              integrated_timing_passed=False, bit_generated=False, board_tested=False))
    print("Synthesis completed; inspect resource and estimated timing reports", flush=True)


if __name__ == "__main__":
    main()
