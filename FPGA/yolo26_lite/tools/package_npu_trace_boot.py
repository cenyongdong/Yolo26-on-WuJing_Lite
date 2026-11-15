"""封装通过完整实现检查的候选 BOOT，并逐字节核验非 PL 区域。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

from common import run_logged, save_json, sha256
from qualify_npu_trace import qualify, require
from qualify_soc import qualify as qualify_soc


ROOT = Path(__file__).resolve().parents[3]


def partition_review(baseline, candidate, payload):
    # 这是当前经过原厂封装／板测的 FMQL 分区契约，不用于其它 BOOT 格式。
    start, size, image_size = 186816, 5980464, 7023552
    end = start + size
    require(len(baseline) == len(candidate) == image_size, "BOOT partition layout changed")
    require(len(payload) == size - 4, "Unexpected PL payload size")
    require(baseline[:start] == candidate[:start] and baseline[end:] == candidate[end:],
            "Firmware/header bytes outside the PL partition changed")
    swapped = b"".join(candidate[index:index + 4][::-1] for index in range(start, end, 4))
    require(swapped[:len(payload)] == payload, "BOOT PL payload does not match the adapted bitstream")
    require(candidate[start + len(payload):end] == baseline[start + len(payload):end], "PL padding changed")
    require(candidate != baseline, "Expected a distinct candidate BOOT")
    return dict(PL_offset=start, PL_bytes=size, all_bytes_outside_PL_identical=True,
                PL_payload_matches_candidate_after_word_swap=True, padding_identical=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--baseline-boot", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--procise", type=Path, default=Path("C:/FudanMicro/Procise"))
    parser.add_argument("--output", type=Path, help="Fresh directory; preserve an earlier failed package")
    parser.add_argument("--profile", choices=("trace", "rectangular-video"), default="trace")
    args = parser.parse_args()
    work, procise = args.build.resolve(strict=True), args.procise.resolve(strict=True)
    qualification = qualify_soc(work) if args.profile == "rectangular-video" else qualify(work)
    for name in ("bit-closure", "procise"):
        row = json.loads((work / "logs" / (name + ".json")).read_text())
        require(row.get("state") == "completed" and row.get("exit_code") == 0, "Native stage failed: " + name)
    baseline = args.baseline_boot.resolve(strict=True)
    require(sha256(baseline) == args.baseline_sha256, "Baseline BOOT identity differs")
    raw = work / "fpai_demo_vivado.runs/impl_pose_filter/ai7030_edif_top.bit"
    adapted = raw.with_name("ai7030_edif_top_disable_icap.bit")
    spec = importlib.util.spec_from_file_location("boot_parser", ROOT / "tools/pose-v1/inspect_selected_boot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload, header = module.bit_payload(adapted.read_bytes())
    output = args.output.resolve() if args.output else work / "boot-package"
    output.mkdir(exist_ok=False)
    # FMQL 镜像头包含位流的文件名。沿用当前合格 BOOT 的封装名称，避免非 PL 元数据变化；
    # 只复制本次适配产物，不修改其内部 header 或 payload，更不补丁修改已生成 BOOT。
    staged_bit = output / "ai7030_top_disable_icap.bit"
    shutil.copyfile(adapted, staged_bit)
    bif = output / "candidate.bif"
    source_bif = (work / "candidate.bif").read_text(encoding="utf-8")
    require(source_bif.count(adapted.as_posix()) == 1, "Prepared BIF does not identify the adapted PL image")
    bif.write_text(source_bif.replace(adapted.as_posix(), staged_bit.as_posix()), encoding="utf-8")
    boot = output / "BOOT.BIN"
    script = output / "boot-package.tcl"
    script.write_text("if {[catch {\nset_device JFMQL30TAI676H\ncreate_boot_image {" +
                      bif.as_posix() + "} {" + boot.as_posix() +
                      "} -fmql\n} result]} {puts $result; puts $::errorInfo; exit 1}\nexit\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(PROCISE_DIR=str(procise), TCL_LIBRARY=str(procise / "tcl8.4"),
               ICTIME_HOME=str(procise), APP_DIR=str(procise), FMSH_DB=str(procise / "db"))
    env["PATH"] = str(procise / "dll") + ";" + str(procise / "bin") + ";" + env["PATH"]
    run_logged([procise / "bin/procise.exe", script], output, "boot-package", env)
    review = partition_review(baseline.read_bytes(), boot.read_bytes(), payload)
    review.update(schema=1, baseline_BOOT_sha256=args.baseline_sha256, BOOT_sha256=sha256(boot),
                  hardware_profile=qualification["hardware_profile"],
                  BOOT_path=str(boot), bytes=boot.stat().st_size,
                  raw_bit=dict(path=str(raw), sha256=sha256(raw)),
                  ICAP_bit=dict(path=str(adapted), sha256=sha256(adapted), header=header),
                  staged_bit=dict(path=str(staged_bit), sha256=sha256(staged_bit)),
                  package_bif=dict(path=str(bif), sha256=sha256(bif)),
                  firmware_inputs={name: sha256(work / "inputs/FPGA/boot" / name) for name in
                                   ("FSBL251210_ddr400_demo.out", "bl31.elf", "u-boot")},
                  deployed=False, rebooted=False)
    save_json(output / "package-review.json", review)
    save_json(output / "qualification.json", qualification)
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
