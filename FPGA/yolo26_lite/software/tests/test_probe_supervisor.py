"""真实 Linux fork/flock/信号回归，工作进程完全模拟；不运行启动预检或任何硬件接口。"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from probe_contract import MODEL, REQUIRED, digest, validate_manifest
EVIDENCE = None


def run_supervisor(directory, descriptor):
    import probe_board
    root = Path(directory)
    # 仅替换外部设备/模型身份边界；被测 supervise、锁继承、子进程及停止逻辑是生产实现。
    probe_board.ROOT = root
    probe_board.LOCK = root / "test.lock"
    probe_board.ACTIVE = root / "test-active.json"
    probe_board.boot_id = lambda: "synthetic-boot"
    probe_board.owners = lambda: []
    probe_board.verify_package = lambda path, manifest: validate_manifest(manifest)
    probe_board.stop_requested = False
    with (root / "supervisor-test.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        probe_board.supervise(descriptor, "synthetic-boot")


@unittest.skipUnless(sys.platform == "linux", "requires Linux fork and flock; no device access")
class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = None
        if EVIDENCE is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="y26-synthetic-supervisor-")
            self.directory = Path(self.temporary.name)
        else:
            self.directory = EVIDENCE / self._testMethodName
            self.directory.mkdir()
        self.process = None

    def tearDown(self):
        if self.process is not None:
            # 仅此测试创建的模拟 supervisor 会被强制结束；它永远不拥有真实设备对象。
            # 正常分支应自行结束，unknown_exit 分支有意检查无限保留，再由测试收回这个假进程。
            (self.directory / "test-release").touch()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.assertFalse(self.process.is_alive(), "synthetic supervisor survived test cleanup")
        if self.temporary is not None:
            self.temporary.cleanup()

    def wait_file(self, name):
        path = self.directory / name
        deadline = time.monotonic() + 10
        while True:
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # 证据文件可能尚在 flush；等待完整 JSON，不把部分写入当作业务失败。
            if time.monotonic() >= deadline:
                self.fail("missing synthetic evidence: " + name)
            time.sleep(0.01)

    def start(self, mode="normal"):
        import fcntl
        root = self.directory
        (root / "synthetic-only.json").write_text(json.dumps(dict(no_hardware=True, mode=mode)))
        manifest = dict(schema=1, source="file_replay", iterations=3, sdk_starts=1, camera_starts=0,
                         boot_writes=0, reboots=0, boot_sha256="a" * 64,
                         files={name: "b" * 64 for name in REQUIRED})
        manifest["files"].update(MODEL)
        (root / "contract.json").write_text(json.dumps(manifest))
        (root / "preflight.json").write_text(json.dumps(dict(boot_id="synthetic-boot",
            contract_sha256=digest(root / "contract.json"))))
        worker = root / "yolo26_infer_probe"
        worker.write_text("#!" + sys.executable + "\n" +
                          Path(__file__).with_name("probe_fake_worker.py").read_text())
        worker.chmod(0o700)
        with (root / "test.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.process = multiprocessing.get_context("fork").Process(
                target=run_supervisor, args=(str(root), lease.fileno()))
            self.process.start()
        # 父进程的文件引用已经关闭；后续持锁只能来自真实继承链，不能由本测试父进程代持。
        self.wait_file("worker-ready.json")
        self.assert_locked(True)

    def assert_locked(self, expected):
        import fcntl
        with (self.directory / "test.lock").open("a") as competitor:
            try:
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = False
            except BlockingIOError:
                locked = True
        self.assertEqual(locked, expected)

    def completed(self):
        outcome = self.wait_file("outcome.json")
        self.process.join(timeout=3)
        self.assertEqual(self.process.exitcode, 0)
        self.assert_locked(False)
        self.assertTrue((self.directory / "test-active.json").is_file(), "startup consumption was erased")
        return outcome

    def test_inherited_lock_outlives_launch_parent(self):
        self.start()
        (self.directory / "test-release").touch()
        outcome = self.completed()
        self.assertEqual((outcome["native_exit"], outcome["completed"]), (0, 3))

    def test_term_requests_native_stop_before_releasing_lock(self):
        self.start()
        os.kill(self.process.pid, signal.SIGTERM)
        outcome = self.completed()
        self.assertEqual(outcome["completed"], 0)
        complete = self.wait_file("native-complete.json")
        self.assertTrue(complete["interrupted"])

    def test_failure_with_shutdown_keeps_original_exit(self):
        self.start("failed_closed")
        (self.directory / "test-release").touch()
        outcome = self.completed()
        self.assertEqual(outcome["native_exit"], 3)
        self.assertFalse(outcome["numerical_pass"])
        self.assertFalse(outcome["automatic_retry"])

    def test_exit_without_shutdown_retains_supervisor_and_lock(self):
        self.start("unknown_exit")
        (self.directory / "test-release").touch()
        retained = self.wait_file("supervisor-retained.json")
        self.assertTrue(retained["lease_retained"])
        self.assertTrue(self.process.is_alive())
        self.assert_locked(True)
        self.assertFalse((self.directory / "outcome.json").exists())
        self.assertEqual(self.wait_file("native-exit.json")["native_exit"], 7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path)
    args, remaining = parser.parse_known_args()
    EVIDENCE = args.evidence_dir
    if EVIDENCE is not None:
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
