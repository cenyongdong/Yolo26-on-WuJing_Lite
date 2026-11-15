"""复制已封存的构建，从原厂 pre-opt hook 之前的合格链接检查点恢复；不改写原失败。"""
import argparse
import json
from pathlib import Path
import re
import shutil

from common import sha256
from prepare_soc import validate_build_location


def prepare_resume(source, output):
    validate_build_location(output)
    if output.exists():
        raise FileExistsError("Recovery needs a fresh directory")
    manifest = json.loads((source / "current-source-manifest.json").read_text())
    outcome = json.loads((source / "logs/opt-banked.json").read_text())
    checkpoint = source / "pipeline-linked-banked.dcp"
    if outcome.get("state") != "completed" or outcome.get("exit_code") != 1 or not checkpoint.is_file():
        raise RuntimeError("Expected completed pre-opt failure and linked checkpoint are absent")
    log = (source / "logs/opt-banked.stdout.log").read_text(errors="replace")
    if "JFM_GUARD_UNPLANNED_INSTALL_WRITE copy -force" not in log or "Start pre_opt_design_patch" in log:
        raise RuntimeError("Failure is not the known pre-hook build-location failure")
    for item in manifest["sources"]:
        if sha256(source / item["input"]) != item["sha256"]:
            raise RuntimeError("Original staged input changed: " + item["input"])
    checkpoint_hash = sha256(checkpoint)
    shutil.copytree(source, output)
    # 日志、报告和 DCP 原样保留；只迁移工具输入中的原构建绝对路径。
    text_suffixes = {".tcl", ".xpr", ".xci", ".xml", ".bif", ".bat", ".cfg"}
    relocated = []
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        original = path.read_bytes()
        try:
            text = original.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        replaced = re.sub(re.escape(source.as_posix()), lambda _: output.as_posix(), text, flags=re.I)
        if replaced != text:
            path.write_text(replaced, encoding="utf-8")
            relocated.append(path.relative_to(output).as_posix())
    if sha256(output / checkpoint.name) != checkpoint_hash:
        raise RuntimeError("Copied checkpoint identity differs")
    plan_path = output / "build-plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(build_root=str(output), recovered_from=str(source))
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for item in manifest["sources"]:
        item["sha256"] = sha256(output / item["input"])
    script = (output / "opt-banked.tcl").read_text(encoding="utf-8-sig")
    anchor = "source {" + (output / "inputs/FPGA/constraints/implementation_opt_pre.tcl").as_posix() + "}"
    if script.count(anchor) != 1:
        raise RuntimeError("Qualified resume boundary is missing or ambiguous")
    resume = output / "opt-resume-r1.tcl"
    resume.write_text("if {[catch {\nset_param general.maxThreads 2\nopen_checkpoint {" +
                      (output / checkpoint.name).as_posix() + "}\n" + script[script.index(anchor):], encoding="utf-8")
    manifest["sources"].append(dict(source="generated:yolo26_resume", input=resume.name, sha256=sha256(resume)))
    (output / "current-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record = dict(original_directory=str(source), original_native_exit=1, original_failure_preserved=True,
                  checkpoint_sha256=checkpoint_hash, checkpoint_reused=True, synthesis_repeated=False,
                  resumed_stage="opt-resume-r1", relocated_inputs=relocated, tool_guard_changed=False,
                  device_operations=0)
    (output / "recovery.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), checkpoint_sha256=checkpoint_hash, relocated=len(relocated))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_resume(args.source.resolve(strict=True), args.output.resolve())
