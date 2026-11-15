"""一次性 NPU 文件回放/摄像头闭环控制器；不写 BOOT、不重启、不超时杀进程、不重试。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import time

from probe_contract import (LIBRARIES, digest, native_launch, require, valid_video_node, validate_completion,
                            validate_live_completion, validate_sequence_completion, verify_package)


ROOT = Path(__file__).resolve().parent
LOCK = Path("/run/pose180-resident.lock")
ACTIVE = Path("/run/pose180-resident.active.json")
stop_requested = False


def stop_handler(signum, frame):
    global stop_requested
    stop_requested = True


def read_json(path):
    return json.loads(path.read_text())


def save_new(path, value):
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def start_ticks(pid):
    try:
        return Path("/proc/{}/stat".format(pid)).read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owners():
    found = []
    for fd in Path("/proc").glob("[0-9]*/fd/*"):
        try:
            name = os.readlink(str(fd))
            if name.startswith(("/dev/axi", "/dev/udmabuf", "/dev/u-dma", "/dev/mem", "/dev/video")):
                found.append(dict(fd=str(fd), target=name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def inherited_lease(descriptor):
    owned, expected = os.fstat(descriptor), LOCK.stat()
    require(descriptor >= 3 and stat.S_ISREG(owned.st_mode) and
            (owned.st_dev, owned.st_ino) == (expected.st_dev, expected.st_ino), "Wrong inherited lease")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def query_uvc_capture(device):
    # Linux videodev2.h 的 VIDIOC_QUERYCAP 和 v4l2_capability，不启流或申请缓冲。
    capability = bytearray(104)
    descriptor = os.open(str(device), os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        fcntl.ioctl(descriptor, 0x80685600, capability)
    finally:
        os.close(descriptor)
    driver = bytes(capability[:16]).split(b"\0", 1)[0]
    caps, device_caps = struct.unpack_from("=II", capability, 84)
    effective = device_caps if caps & 0x80000000 else caps
    if driver == b"uvcvideo" and effective & 0x04000001 == 0x04000001:
        return bytes(capability[48:80]).split(b"\0", 1)[0].decode("ascii")
    return None


def select_usb_camera(usb_id):
    candidates = []
    for device in sorted(Path("/dev").glob("video*")):
        if not valid_video_node(str(device)):
            continue
        sysfs = Path("/sys/class/video4linux") / device.name / "device"
        if not sysfs.exists():
            continue
        interface = sysfs.resolve(strict=True)
        for parent in (interface, *interface.parents):
            vendor, product = parent / "idVendor", parent / "idProduct"
            if vendor.is_file() and product.is_file():
                if vendor.read_text().strip() + ":" + product.read_text().strip() == usb_id:
                    bus = query_uvc_capture(device)
                    if bus:
                        candidates.append(dict(device=str(device), usb_id=usb_id, bus_info=bus))
                break
    require(len(candidates) == 1, "USB camera must identify one streaming UVC video capture node")
    return candidates[0]


def preflight(expected_boot):
    # BOOT 只读审计沿用既有 FAT16 实现，不复制解析器、不 mount 或改启动分区。
    from board_probe import audit_fat16
    require(ROOT.parent == Path("/root") and ROOT.name.startswith("yolo26-"), "Wrong board package root")
    require(expected_boot == boot_id(), "BOOT changed before launch")
    require(not ACTIVE.exists(), "This startup has already been consumed; maintenance handoff is required")
    require(not owners(), "Another board/camera owner is present")
    manifest = read_json(ROOT / "contract.json")
    verify_package(ROOT, manifest)
    partition = audit_fat16("/dev/mmcblk0p1")
    boot_files = [item for item in partition["files"] if item["name"].upper() == "BOOT.BIN"]
    require(len(boot_files) == 1 and boot_files[0]["sha256"] == manifest["boot_sha256"],
            "Installed BOOT file does not match the independently qualified image")
    libraries = {name: digest(Path("/usr/lib/aarch64-linux-gnu") / name) for name in LIBRARIES}
    require(libraries == LIBRARIES, "Installed SDK libraries differ")
    camera = {"video_camera": select_usb_camera(manifest["camera_usb_id"])} if manifest["schema"] == 4 else {}
    require(boot_id() == expected_boot and not ACTIVE.exists() and not owners(), "State changed during audit")
    return dict(boot_id=expected_boot, boot_file=boot_files[0], libraries=libraries,
                device_owners=[], contract_sha256=digest(ROOT / "contract.json"), sdk_starts=0, **camera)


def launch(expected_boot):
    with LOCK.open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = preflight(expected_boot)
        save_new(ROOT / "preflight.json", checked)
        with (ROOT / "supervisor.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "probe_board.py"),
                                        "supervise", "--lease-fd", str(lease.fileno()),
                                        "--expected-boot", expected_boot], cwd=str(ROOT),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(lease.fileno(),), start_new_session=True)
        record = dict(pid=process.pid, start_ticks=start_ticks(process.pid), boot_id=expected_boot)
        save_new(ROOT / "supervisor-process.json", record)
        # 仅关闭父进程的引用，不 LOCK_UN；子进程继承同一 open-file description，继续拥有独占。
        return record


def retain(reason):
    try:
        save_new(ROOT / "supervisor-retained.json", dict(reason=str(reason), boot_id=boot_id(),
                 pid=os.getpid(), lease_retained=True, automatic_retry=False))
    except Exception:
        pass
    print("Unproved worker shutdown; retaining inherited lease: " + str(reason), flush=True)
    while True:
        time.sleep(1)


def supervise(descriptor, expected_boot):
    inherited_lease(descriptor)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_handler)
    worker = None
    consumed = False
    try:
        require(boot_id() == expected_boot and not ACTIVE.exists() and not owners(), "Launch state changed")
        manifest = read_json(ROOT / "contract.json")
        budget = verify_package(ROOT, manifest)
        prior = read_json(ROOT / "preflight.json")
        require(prior["boot_id"] == expected_boot and prior["contract_sha256"] == digest(ROOT / "contract.json"),
                "Preflight does not belong to this immutable run")
        command, permit = native_launch(ROOT, descriptor, expected_boot, manifest, prior.get("video_camera"))
        require(not stop_requested, "Stop requested before consuming startup")
        # 启动登记先于子进程及 SDK；保守消费后绝不删除登记或自动补一次启动。
        save_new(ACTIVE, dict(boot_id=expected_boot, run=str(ROOT), sdk_start_budget=1))
        consumed = True
        with (ROOT / "native.log").open("xb") as log:
            worker = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, pass_fds=(descriptor,))
            ticks = start_ticks(worker.pid)
            require(ticks is not None, "Native worker exited before its identity was recorded")
            save_new(ROOT / "worker-process.json", dict(pid=worker.pid, start_ticks=ticks,
                     boot_id=expected_boot, supervisor_pid=os.getpid(), command=command))
            temporary = ROOT / "startup-permit.pending.json"
            save_new(temporary, permit)
            # 同目录硬链接发布完整许可；目标已存在时拒绝，不覆盖半写入或旧许可。
            os.link(str(temporary), str(ROOT / "startup-permit.json"))
            temporary.unlink()
            signal_sent = False
            while worker.poll() is None:
                if stop_requested and not signal_sent and start_ticks(worker.pid) == ticks:
                    worker.send_signal(signal.SIGTERM)  # C++ 处理器只请求停止；不取消在途硬件。
                    signal_sent = True
                time.sleep(0.1)
            native_exit = worker.wait()
        save_new(ROOT / "native-exit.json", dict(boot_id=expected_boot, native_exit=native_exit,
                 pid=worker.pid, start_ticks=ticks, automatic_retry=False))
        failed_closed = ROOT / "native-failed-closed.json"
        if native_exit != 0 and failed_closed.is_file():
            closed = read_json(failed_closed)
            require(closed.get("boot_id") == expected_boot and closed.get("native_shutdown_complete") is True and
                    closed.get("allocation_count_after") == 0 and closed.get("original_failure_preserved") is True,
                    "Failure cleanup is not proven")
            require(boot_id() == expected_boot and not owners(), "Post-failure device ownership differs")
            save_new(ROOT / "outcome.json", dict(closed, native_exit=native_exit, device_owners=[],
                     numerical_pass=False, automatic_retry=False))
            print("Original failure retained; native shutdown proven, no retry", flush=True)
            return
        completion = read_json(ROOT / "native-complete.json")
        live_evidence = {}
        if manifest["source"] == "live_camera":
            summary = read_json(ROOT / "native-live-summary.json")
            completed = validate_live_completion(completion, expected_boot, manifest, native_exit,
                                                 summary)
            if manifest["schema"] == 4:
                live_evidence = {key: summary[key] for key in ("complete_chain_30fps_10min_passed",
                                 "adopted_reference_matches", "adopted_reference_boxes")}
        elif manifest["source"] == "file_sequence":
            completed = validate_sequence_completion(completion, expected_boot, manifest, native_exit,
                                                     read_json(ROOT / "native-sequence-summary.json"))
        else:
            completed = validate_completion(completion, expected_boot, budget, native_exit)
        require(boot_id() == expected_boot and not owners(), "Post-run device ownership or BOOT differs")
        save_new(ROOT / "outcome.json", dict(boot_id=expected_boot, native_exit=native_exit,
                 completed=completed, native_shutdown_complete=True, allocation_count_after=0,
                 sdk_starts=1, capture_starts=manifest["camera_starts"], boot_writes=0, reboots=0, device_owners=[],
                 source=manifest["source"], camera_30fps_claim=completion["camera_30fps_claim"], **live_evidence))
        print("Finite NPU probe closed with native completion proof", flush=True)
    except Exception as error:
        # 原生对象可能仍活着；即使进程已异常退出，也不能推定硬件排空。
        if consumed or worker is not None:
            retain(error)
        raise


def inspect():
    records = {}
    for name in ("preflight", "supervisor-process", "worker-process", "native-exit", "native-complete",
                 "native-failure", "native-failed-closed", "native-retained", "supervisor-retained", "outcome"):
        path = ROOT / (name + ".json")
        if path.is_file():
            records[name] = read_json(path)
    return dict(boot_id=boot_id(), owners=owners(), active=read_json(ACTIVE) if ACTIVE.exists() else None,
                records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("launch", "inspect", "supervise"))
    parser.add_argument("--expected-boot")
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    if args.stage == "inspect":
        result = inspect()
    else:
        require(args.expected_boot is not None, "Current expected BOOT identity is required")
        if args.stage == "launch":
            result = launch(args.expected_boot)
        else:
            require(args.lease_fd is not None, "Supervisor requires an inherited lease")
            supervise(args.lease_fd, args.expected_boot)
            result = dict(supervisor_closed=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
