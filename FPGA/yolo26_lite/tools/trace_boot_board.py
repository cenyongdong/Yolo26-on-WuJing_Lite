"""临时观察 BOOT 的有限维护；复用合格 FAT 操作，始终持有共享锁，不初始化 SDK。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from probe_board import owners, read_json, save_new, start_ticks
from probe_contract import digest, require, verify_video_review


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "boot"))
import board_support as boot


def closed_startup(expected):
    """没有设备 fd 不能替代前一批的原生完成；记录与进程身份必须同时成立。"""
    require(boot.boot_id() == expected, "Startup changed during maintenance")
    require(not owners() and not boot.owners(), "Another device owner is present")
    active_path = Path("/run/pose180-resident.active.json")
    if not active_path.exists():
        return dict(boot_id=expected, prior_run=None, startup_not_consumed=True)
    active = read_json(active_path)
    prior = Path(active["run"])
    require(active["boot_id"] == expected and prior.parent == Path("/root") and
            prior.name.startswith("yolo26-") and not prior.is_symlink(), "Unreviewed previous startup owner")
    outcome = read_json(prior / "outcome.json")
    require(outcome["boot_id"] == expected and type(outcome["native_exit"]) is int and
            outcome["native_shutdown_complete"] is True and outcome["allocation_count_after"] == 0,
            "Previous batch lacks complete native shutdown")
    if outcome["native_exit"] != 0:
        # 失败结果不可当作性能通过；但已证明安全关闭时仍须能够恢复原 BOOT。
        # 不放行只有进程退出、没有原生排空证据的失败负载。
        closed = read_json(prior / "native-failed-closed.json")
        native = read_json(prior / "native-exit.json")
        require(closed.get("boot_id") == native.get("boot_id") == expected and
                native.get("native_exit") == outcome["native_exit"] and
                closed.get("native_shutdown_complete") is True and closed.get("allocation_count_after") == 0 and
                closed.get("original_failure_preserved") is True and outcome.get("original_failure_preserved") is True,
                "Failed batch lacks independently recorded native close and exit evidence")
    for name in ("worker-process", "supervisor-process"):
        process = read_json(prior / (name + ".json"))
        require(str(start_ticks(process["pid"])) != str(process["start_ticks"]), "Previous batch still running")
    return dict(boot_id=expected, prior_run=str(prior), prior_shutdown=outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "install", "fallback", "reboot"))
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--boot-sha256", required=True)
    args = parser.parse_args()
    require(args.label and all(c.isalnum() or c in "-_" for c in args.label), "Invalid operation label")
    is_video = ROOT.name.startswith("yolo26-video-boot-")
    require(ROOT.parent == Path("/root") and (is_video or ROOT.name.startswith("yolo26-trace-boot-")),
            "Wrong maintenance root")
    contract = read_json(ROOT / "maintenance-contract.json")
    writes = (contract["boot_installs"], contract["boot_fallbacks"])
    restored_startup_only = is_video and writes == (0, 0) and contract["reboots"] == 1
    require(contract["schema"] == 1 and 1 <= contract["reboots"] <= 6 and
            (writes == (1, 1) or restored_startup_only), "Unexpected maintenance budget")
    for name, expected in contract["files"].items():
        require(digest(ROOT / name) == expected, "Maintenance payload identity changed: " + name)
    qualification = read_json(ROOT / "boot/qualification.json")
    review = read_json(ROOT / "boot/package-review.json")
    require(qualification["bit_generation_prerequisite_pass"], "Incomplete hardware qualification")
    baseline, candidate = review["baseline_BOOT_sha256"], review["BOOT_sha256"]
    if is_video:
        verify_video_review({**qualification, **review}, {"boot_sha256": candidate})
    require(args.boot_sha256 in (baseline, candidate), "Unknown BOOT identity")
    require(not restored_startup_only or (args.action in ("audit", "reboot") and args.boot_sha256 == baseline),
            "Restored startup batch only audits or boots the original BOOT")
    boot.ROOT, boot.PACKAGE = ROOT, ROOT / "boot"
    boot.META, boot.MOUNT = ROOT / "boot-operations", ROOT / "boot-mount"
    boot.META.mkdir(exist_ok=True)
    boot.MOUNT.mkdir(exist_ok=True)
    boot.CFG["package_id"] = ROOT.name
    boot.identity_record = lambda value: dict(value, package_id=ROOT.name)
    with Path("/run/pose180-resident.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        closed = closed_startup(args.expected_boot)
        audit = boot.audit(args.boot_sha256, args.expected_boot)
        if args.action == "audit":
            save_new(boot.META / (args.label + "-audit.json"), dict(audit=audit, prior=closed))
            print(json.dumps(dict(boot_id=args.expected_boot, BOOT_sha256=args.boot_sha256,
                                 owners=[], prior_run=closed["prior_run"], audit_pass=True)))
            return
        if args.action in ("install", "fallback"):
            before, after = (baseline, candidate) if args.action == "install" else (candidate, baseline)
            require(args.boot_sha256 == before, "Unexpected pre-operation BOOT")
            source = ROOT / "boot/BOOT.BIN" if args.action == "install" else boot.META / "BOOT-before.readback.bin"
            if args.action == "install":
                boot.fat_files(boot.META / "BOOT-before.readback.bin")
            require(digest(boot.META / "BOOT-before.readback.bin") == baseline, "Actual backup identity differs")
            save_new(boot.META / (args.action + "-prior.json"), closed)
            # replace_boot 在写入前独占预留本动作，任何失败不会自动补写或删除原记录。
            result = boot.replace_boot(source, before, after, args.action, args.action + ".json")
            print(json.dumps({key: result[key] for key in ("passed", "boot_id", "BOOT_sha256",
                                                         "boot_partition_unmounted")}))
            return
        attempts = list(boot.META.glob("reboot-*.attempt.json"))
        require(len(attempts) < contract["reboots"], "Maintenance reboot budget consumed")
        name = "reboot-" + args.label
        boot.reserve(name, dict(BOOT_sha256=args.boot_sha256, prior_shutdown=closed))
        os.sync()
        # 正常系统关机；断连不是成功的新启动证明。锁传入请求进程，下一步重新核验 boot_id。
        subprocess.run(["/sbin/shutdown", "-r", "now"], check=True, pass_fds=(lease.fileno(),))
        print(json.dumps(dict(reboot_requested=True, boot_id=args.expected_boot,
                             BOOT_sha256=args.boot_sha256)), flush=True)


if __name__ == "__main__":
    main()
