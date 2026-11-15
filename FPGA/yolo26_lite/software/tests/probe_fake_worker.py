"""仅供监督器进程测试的模拟工作进程；从不导入 SDK，也不访问 /dev 或板卡锁。"""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


stopped = False


def stop_handler(signum, frame):
    global stopped
    stopped = True


def main():
    descriptor, boot, directory, iterations = int(sys.argv[1]), sys.argv[2], Path(sys.argv[6]), int(sys.argv[7])
    # 每个测试先创建这个明确的模拟标记，禁止把此脚本误放进正常包中当作 NPU 工作进程。
    scenario = json.loads((directory / "synthetic-only.json").read_text())
    assert scenario["no_hardware"] is True
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop_handler)
    deadline = time.monotonic() + 8
    while not (directory / "startup-permit.json").exists():
        assert time.monotonic() < deadline, "test permit timeout"
        time.sleep(0.01)
    permit = json.loads((directory / "startup-permit.json").read_text())
    assert permit["iterations"] == iterations and permit["boot_id"] == boot
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owned = os.fstat(descriptor)
    expected = (directory / "test.lock").stat()
    assert (owned.st_dev, owned.st_ino) == (expected.st_dev, expected.st_ino)
    (directory / "worker-ready.json").write_text(json.dumps(dict(pid=os.getpid(), synthetic=True)))
    while not stopped and not (directory / "test-release").exists():
        assert time.monotonic() < deadline, "test release timeout"
        time.sleep(0.01)
    closed = dict(boot_id=boot, native_shutdown_complete=True, allocation_count_after=0)
    mode = scenario["mode"]
    if mode == "failed_closed":
        closed.update(original_failure_preserved=True, automatic_retry=False)
        (directory / "native-failed-closed.json").write_text(json.dumps(closed))
        return 3
    if mode == "unknown_exit":
        return 7
    closed.update(completed=0 if stopped else iterations, interrupted=stopped, sdk_starts=1,
                  capture_starts=0, boot_writes=0, reboots=0, camera_30fps_claim=False)
    (directory / "native-complete.json").write_text(json.dumps(closed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
