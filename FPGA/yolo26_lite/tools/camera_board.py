"""板端一次性 UVC 构建/采集控制器。共享锁覆盖子进程全生命周期；不操作 SDK、BOOT 或重启。"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import time

from benchmark_contract import validate_cpu_receipt

ROOT = Path(__file__).resolve().parent
LOCK = Path("/run/pose180-resident.lock")


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owners():
    result = []
    for fd in Path("/proc").glob("[0-9]*/fd/*"):
        try:
            target = os.readlink(str(fd))
            if target.startswith(("/dev/axi", "/dev/udmabuf", "/dev/u-dma", "/dev/mem", "/dev/video")):
                result.append(dict(fd=str(fd), target=target))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def verify_idle():
    current_boot = boot_id()
    if owners():
        raise RuntimeError("Shared board or camera still has a device owner")
    active_path = Path("/run/pose180-resident.active.json")
    if not active_path.exists():
        return dict(boot_id=current_boot, startup_consumed=False, device_owners=[])
    active = json.loads(active_path.read_text())
    if active.get("boot_id") != current_boot:
        raise RuntimeError("Active-run identity does not match this startup")
    prior = Path(active["run"]).resolve()
    if Path("/root") not in prior.parents:
        raise RuntimeError("Prior run is outside the known board workspace")
    recovery = prior / "recovery-closed.json"
    outcome_path = recovery if recovery.is_file() else prior / "outcome.json"
    outcome = json.loads(outcome_path.read_text())
    process = json.loads((prior / "worker-process.json").read_text())
    valid_exit = outcome.get("native_exit") == 0
    if recovery.is_file():
        valid_exit = (outcome.get("original_failure_preserved") and outcome.get("lease_free") and
                      not outcome.get("device_owners") and (prior / "failure-before-cleanup.json").is_file())
    elif not valid_exit and prior.name.startswith("yolo26-"):
        closed = json.loads((prior / "native-failed-closed.json").read_text())
        native = json.loads((prior / "native-exit.json").read_text())
        valid_exit = (closed.get("boot_id") == native.get("boot_id") == current_boot and
                      closed.get("native_shutdown_complete") is True and closed.get("allocation_count_after") == 0 and
                      closed.get("original_failure_preserved") is True and outcome.get("original_failure_preserved") is True and
                      native.get("native_exit") == outcome.get("native_exit") and type(native.get("native_exit")) is int and
                      native.get("native_exit") != 0 and native.get("pid") == process.get("pid") and
                      native.get("start_ticks") == process.get("start_ticks"))
    if not (outcome.get("boot_id") == current_boot and outcome.get("native_shutdown_complete") and
            outcome.get("allocation_count_after") == 0 and valid_exit):
        raise RuntimeError("Previous hardware shutdown is not proven")
    stat = Path("/proc/{}/stat".format(process["pid"]))
    if stat.exists() and stat.read_text().split(") ", 1)[1].split()[19] == str(process["start_ticks"]):
        raise RuntimeError("Previous native worker still exists")
    return dict(boot_id=current_boot, startup_consumed=True, prior_outcome=str(outcome_path),
                prior_shutdown_proven=True, device_owners=[])


def verify_sources():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source package identity mismatch")
    return manifest


def execute(label, command, lease, address_limit_mb=0):
    with (ROOT / (label + ".log")).open("xb") as log:
        # 子进程继承同一个 flock 文件描述，SSH/控制器意外退出也不能提前释放仍在运行的负载租约。
        # SDK 头文件编译只限制本批编译器地址空间；超限按原生失败保留，不终止其它任务或重试。
        def limit_address_space():
            budget = address_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
        process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),),
                                   preexec_fn=limit_address_space if address_limit_mb else None)
        stat = Path("/proc/{}/stat".format(process.pid))
        ticks = stat.read_text().split(") ", 1)[1].split()[19] if stat.exists() else None
        save_new(ROOT / (label + ".process.json"), dict(pid=process.pid, start_ticks=ticks,
                 boot_id=boot_id(), command=command, started_ns=time.monotonic_ns()))
        code = process.wait()
    result = dict(command=command, native_exit=code, child_returned=True)
    save_new(ROOT / (label + ".result.json"), result)
    if code:
        raise RuntimeError(label + " failed; original exit and log retained, no automatic retry")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "capture", "capture-modes", "cpu", "tail", "compose", "status"))
    parser.add_argument("--compiler-memory-mib", type=int, choices=(512, 600), default=512,
                        help="Bound SDK compiler virtual address space; preserve at least 128 MiB available")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sdk", action="store_true", help="Compile/link SDK targets only; never execute them")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600 or not 0 <= args.warmup <= 30:
        parser.error("Invalid finite capture duration")
    if not 8 <= args.iterations <= 2000:
        parser.error("Invalid finite CPU benchmark iteration count")
    if args.sdk and args.stage != "build":
        parser.error("--sdk only selects compile/link targets, not an execution stage")
    if args.compiler_memory_mib != 512 and not (args.sdk and args.stage == "build"):
        parser.error("Compiler memory override only applies to SDK compilation")
    if args.stage == "status":
        print(json.dumps(dict(boot_id=boot_id(), owners=owners(), files=[p.name for p in ROOT.glob("*.json")]), indent=2))
        return
    verify_sources()
    with LOCK.open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = verify_idle()
        save_new(ROOT / (args.stage + "-preflight.json"), before)
        capture_starts = 0
        if args.stage == "build":
            configure = ["cmake", "-S", str(ROOT), "-B", str(ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"]
            if args.sdk:
                memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available_kib = int(memory["MemAvailable"].split()[0])
                if available_kib < max(660, args.compiler_memory_mib + 128) * 1024:
                    raise RuntimeError("Insufficient free board memory for isolated SDK compilation")
                for comm in Path("/proc").glob("[0-9]*/comm"):
                    try:
                        if comm.read_text().strip() in ("cc1plus", "cc1", "cmake"):
                            raise RuntimeError("Another board compilation is active")
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                configure += ["-DYOLO26_BUILD_SDK=ON", "-DCMAKE_PREFIX_PATH=/usr",
                              "-DYOLO26_RELEASE_GUARD_DIR=" + str(ROOT / "support")]
                save_new(ROOT / "compile-budget.json", dict(available_kib=available_kib,
                         compiler_address_limit_mib=args.compiler_memory_mib, parallel_jobs=1, sdk_execution=False))
            execute("configure", configure, lease)
            execute("compile", ["cmake", "--build", str(ROOT / "build"), "--parallel", "1"], lease,
                    address_limit_mb=args.compiler_memory_mib if args.sdk else 0)
            tests = []
            for name in ("postprocess", "frames", "preprocess", "control", "npu_trace", "compose", "metrics",
                         "display_buffers", "display_control", "inference_completion", "video_engine", "capture_color"):
                tests.append(execute("test-" + name, [str(ROOT / "build" / ("test_" + name))], lease))
            if args.sdk:
                # 只分配 HostDevice 内存，与原厂 SFB 导出逐字节比较，不打开 NPU 或 FPGA 设备。
                tests.append(execute("test-icraft_tensor_io", [str(ROOT / "build/test_icraft_tensor_io")], lease))
            save_new(ROOT / "tests.result.json", dict(native_exit=0, tests=tests))
        elif args.stage in ("cpu", "tail", "compose"):
            test_result = json.loads((ROOT / "tests.result.json").read_text())
            if test_result["native_exit"] != 0:
                raise RuntimeError("Native software tests did not pass")
            if args.stage == "cpu":
                execute("cpu", [str(ROOT / "build/yolo26_cpu_benchmark"), str(args.iterations)], lease)
                lines = [json.loads(line) for line in (ROOT / "cpu.log").read_text().splitlines()]
                validate_cpu_receipt(lines, args.iterations)
            elif args.stage == "compose":
                output = ROOT / "compose.rgb565"
                execute("compose", [str(ROOT / "build/compose_fixture"), str(output)], lease)
                if output.stat().st_size != 6 * 1280 * 720 * 2:
                    raise RuntimeError("Complete composition fixture shape differs")
            else:
                # 三组既有离线模型输出，仅检查 CPU 尾处理；每组固定 50 次，完整结果带回主机比较。
                for index in range(3):
                    label = "tail-{}".format(index)
                    output = ROOT / (label + "-detections.f32")
                    if output.exists():
                        raise RuntimeError("Preserve prior tail-processing output")
                    execute(label, [str(ROOT / "build/postprocess_fixture"),
                                    str(ROOT / "fixtures/case-{}.f32".format(index)), str(output)], lease)
                    if output.stat().st_size != 300 * 6 * 4:
                        raise RuntimeError("Tail-processing output shape differs")
        else:
            test_result = json.loads((ROOT / "tests.result.json").read_text())
            if test_result["native_exit"] != 0:
                raise RuntimeError("Native software tests did not pass")
            # 固定三种设备已枚举的模式，一项正常关闭后才开始下一项；任意失败中止，不重试。
            modes = ((640, 480), (320, 240), (160, 120)) if args.stage == "capture-modes" else ((640, 480),)
            for width, height in modes:
                label = "capture-{}x{}".format(width, height) if args.stage == "capture-modes" else "capture"
                frame_name = label + ".yuyv" if args.stage == "capture-modes" else "last-frame.yuyv"
                command = [str(ROOT / "build/yolo26_camera_probe"), "/dev/video1",
                           str(args.seconds), str(args.warmup), str(ROOT / frame_name)]
                if args.stage == "capture-modes":
                    command += [str(width), str(height), "30"]
                execute(label, command, lease)
                capture_starts += 1
                lines = [json.loads(line) for line in (ROOT / (label + ".log")).read_text().splitlines()]
                summary = lines[-1]
                if (summary.get("kind") != "summary" or not summary.get("clean_shutdown") or
                        summary.get("interrupted") or summary.get("pool_available") != 6):
                    raise RuntimeError("Camera cleanup receipt is absent")
                verify_idle()
        after = verify_idle()
        if after["boot_id"] != before["boot_id"]:
            raise RuntimeError("Board restarted during the owned workload")
        save_new(ROOT / (args.stage + "-closed.json"), dict(before=before, after=after, sdk_starts=0,
                 boot_writes=0, reboots=0, capture_starts=capture_starts,
                 lease_held_until_children_returned=True))
        print(json.dumps(dict(stage=args.stage, completed=True, boot_id=after["boot_id"])))


if __name__ == "__main__":
    main()
