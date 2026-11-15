"""在隔离目录运行原厂 XSim；验证原生退出码及 testbench 完成记录。"""
from pathlib import Path
import argparse
import datetime
import json

from common import run_logged, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=("dma", "color", "resize", "pack", "stream", "preprocess", "descriptor",
                                          "mailbox", "registers", "control", "peripheral",
                                          "read_share", "write_share", "display", "display_clock", "soc",
                                          "soc_rectangular", "npu_bus", "npu_trace"), default="dma")
    parser.add_argument("--vivado-bin", type=Path, default=Path("D:/vivado18.3/Vivado/2018.3/bin"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = root / "build" / ("xsim-" + args.test + "-" + stamp)
    output.mkdir(parents=True)
    control_sources = ["rtl/y26_register_bank.sv", "rtl/y26_mailbox.sv", "rtl/y26_command_dispatch.sv",
                       "rtl/y26_control.sv"]
    preprocess_sources = ["rtl/y26_dma_descriptor.sv", "rtl/y26_dma_mm2s.sv", "rtl/y26_color_pipeline.sv",
                          "rtl/y26_pixel_unpack.sv", "rtl/y26_resize_letterbox.sv", "rtl/y26_pixel_pack.sv",
                          "rtl/y26_dma_s2mm.sv", "rtl/y26_stream_register.sv", "rtl/y26_preprocess.sv"]
    display_sources = ["../hdmi_candidate/rtl/" + name + ".v" for name in (
        "hdmi_axi_read", "hdmi_scan_control", "hdmi_command_mailbox", "hdmi_candidate_core",
        "hdmi_register_bank", "hdmi_candidate_register_core", "hdmi_pixel_scan", "hdmi_stream_fifo", "hdmi_scanout")]
    source_names = {"descriptor": ["rtl/y26_dma_descriptor.sv", "tb/tb_descriptor.sv"],
                    "npu_bus": ["rtl/y26_npu_bus_bins.sv", "tb/tb_npu_bus.sv"],
                    "npu_trace": ["rtl/y26_npu_bus_bins.sv", "rtl/y26_npu_trace_store.sv",
                                  "rtl/y26_npu_trace_csr.sv", "rtl/y26_mailbox.sv", "tb/tb_npu_trace.sv"],
                    "stream": ["rtl/y26_stream_register.sv", "tb/tb_stream.sv"],
                    "read_share": ["rtl/y26_axi_read_share.sv", "tb/tb_read_share.sv"],
                    "write_share": ["rtl/y26_axi_write_share.sv", "tb/tb_write_share.sv"],
                    "display": display_sources + ["../hdmi_candidate/sim/tb_stream_fifo.sv",
                               "../hdmi_candidate/sim/tb_scanout.sv", "tb/tb_display.sv"],
                    "display_clock": ["../hdmi_candidate/rtl/integration/hdmi_clock_jfm.v",
                                      "tb/tb_display_clock.sv"],
                    "mailbox": ["rtl/y26_mailbox.sv", "tb/tb_mailbox.sv"],
                    "registers": ["rtl/y26_register_bank.sv", "tb/tb_registers.sv"],
                    "control": control_sources + ["tb/tb_control.sv"],
                    "peripheral": control_sources + preprocess_sources + ["rtl/y26_peripheral.sv",
                                  "tb/axi_memory_model.sv", "tb/tb_peripheral.sv"],
                    "dma": ["rtl/y26_dma_descriptor.sv", "rtl/y26_dma_mm2s.sv", "rtl/y26_dma_s2mm.sv", "tb/tb_dma.sv"],
                    "color": ["rtl/y26_color_pipeline.sv", "rtl/y26_pixel_unpack.sv", "tb/tb_color.sv"],
                    "resize": ["rtl/y26_resize_letterbox.sv", "tb/tb_resize.sv"],
                    "pack": ["rtl/y26_pixel_pack.sv", "tb/tb_pack.sv"],
                    "preprocess": preprocess_sources + ["tb/axi_memory_model.sv", "tb/tb_preprocess.sv"]}
    source_names["soc"] = source_names["peripheral"] + ["rtl/y26_axi_read_share.sv",
                          "rtl/y26_axi_write_share.sv", "rtl/y26_soc_bridge.sv", "tb/y26_soc_test_adapter.sv"]
    source_names["soc_rectangular"] = source_names["soc"]
    sources = [root / name for name in source_names[args.test]]
    if args.test == "color":
        from color_reference import generate
        generate(output)
    if args.test == "resize":
        from resize_reference import generate
        generate(output)
    soc = args.test in ("soc", "soc_rectangular")
    if args.test in ("preprocess", "peripheral") or soc:
        from preprocess_reference import generate
        generate(output, (512, 384) if args.test == "soc_rectangular" else (640, 640))
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    elaboration = ["tb_peripheral" if soc else "tb_" + args.test]
    defines = ["--define", "YOLO26_SOC_TEST"] if soc else []
    if args.test == "soc_rectangular":
        defines += ["--define", "YOLO26_RECT_TEST"]
    if args.test == "display_clock":
        glbl = args.vivado_bin.parent / "data" / "verilog" / "src" / "glbl.v"
        sources.append(glbl)
        source_hashes[str(glbl)] = sha256(glbl)
        elaboration += ["glbl", "-L", "unisims_ver", "-L", "secureip"]
    save_json(output / "sources.json", source_hashes)
    (output / "run.tcl").write_text("run all\nquit\n", encoding="ascii")
    stages = []
    for name, arguments in (("xvlog", ["--sv"] + defines + sources),
                            ("xelab", elaboration + ["-s", "test_snapshot", "-debug", "typical",
                                       "--timescale", "1ns/1ps"]),
                            ("xsim", ["test_snapshot", "-tclbatch", "run.tcl"])):
        stages.append(run_logged([args.vivado_bin / (name + ".bat")] + arguments, output, name))
    result = json.loads((output / "result.json").read_text())
    if result.get("passed") is not True or result.get("cases", 0) == 0:
        raise RuntimeError("Testbench did not pass")
    save_json(root / "build" / (args.test + "-latest.json"), dict(directory=str(output), stages=stages,
              result=result, board_tested=False))
    print(json.dumps(dict(directory=str(output), result=result)), flush=True)


if __name__ == "__main__":
    main()
