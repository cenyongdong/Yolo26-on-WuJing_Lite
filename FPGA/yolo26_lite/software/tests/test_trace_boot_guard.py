"""维护入口必须同时看到原生排空证据和进程退出；无 fd 单独不足以放行。"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
boot = types.SimpleNamespace(boot_id=lambda: "current", owners=lambda: [])
with patch.dict(sys.modules, {"fcntl": types.ModuleType("fcntl"), "board_support": boot}):
    spec = importlib.util.spec_from_file_location("trace_boot_guard_test", TOOLS / "trace_boot_board.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class MaintenanceGuard(unittest.TestCase):
    def test_restored_startup_batch_rejects_writes_and_candidate(self):
        contract = dict(schema=1, boot_installs=0, boot_fallbacks=0, reboots=1, files={})
        files = {"maintenance-contract.json": contract,
                 "qualification.json": {"bit_generation_prerequisite_pass": True},
                 "package-review.json": {"baseline_BOOT_sha256": "original", "BOOT_sha256": "candidate"}}
        with patch.object(module, "ROOT", Path("/root/yolo26-video-boot-restoration")), \
                patch.object(module, "read_json", side_effect=lambda path: files[path.name]), \
                patch.object(module, "verify_video_review"):
            for action, image in (("install", "original"), ("fallback", "candidate"), ("reboot", "candidate")):
                with self.subTest(action=action), patch.object(sys, "argv", ["trace_boot_board.py", action,
                        "--label", "test", "--expected-boot", "current", "--boot-sha256", image]):
                    with self.assertRaisesRegex(ValueError, "only audits or boots the original BOOT"):
                        module.main()

    def test_closed_startup_and_unconsumed_startup(self):
        active = Mock()
        active.exists.return_value = False
        outcome = dict(boot_id="current", native_exit=0, native_shutdown_complete=True, allocation_count_after=0)
        failed_closed, native_exit = {}, {}

        def path(name):
            return active if str(name) == "/run/pose180-resident.active.json" else Path(name)

        def read(name):
            if name == active:
                return dict(boot_id="current", run="/root/yolo26-previous")
            if name.name == "outcome.json":
                return outcome
            if name.name == "native-failed-closed.json":
                return failed_closed
            if name.name == "native-exit.json":
                return native_exit
            return dict(pid=123, start_ticks="old")

        with patch.object(module, "Path", side_effect=path), patch.object(module, "read_json", side_effect=read), \
                patch.object(module, "owners", return_value=[]), patch.object(module, "start_ticks", return_value=None):
            self.assertTrue(module.closed_startup("current")["startup_not_consumed"])
            active.exists.return_value = True
            self.assertEqual(module.closed_startup("current")["prior_run"], str(Path("/root/yolo26-previous")))
            for key, bad in (("native_exit", 1), ("native_shutdown_complete", False),
                             ("allocation_count_after", 1), ("boot_id", "stale")):
                original = outcome[key]
                outcome[key] = bad
                with self.assertRaises(ValueError):
                    module.closed_startup("current")
                outcome[key] = original
            outcome.update(native_exit=1, original_failure_preserved=True)
            failed_closed.update(boot_id="current", native_shutdown_complete=True, allocation_count_after=0,
                                 original_failure_preserved=True)
            native_exit.update(boot_id="current", native_exit=1)
            self.assertEqual(module.closed_startup("current")["prior_shutdown"]["native_exit"], 1)
            for key, bad in (("native_shutdown_complete", False), ("allocation_count_after", 1),
                             ("original_failure_preserved", False), ("boot_id", "stale")):
                original = failed_closed[key]
                failed_closed[key] = bad
                with self.assertRaises(ValueError):
                    module.closed_startup("current")
                failed_closed[key] = original
            native_exit["native_exit"] = 0
            with self.assertRaises(ValueError):
                module.closed_startup("current")
            outcome["native_exit"] = 0
            with patch.object(module, "start_ticks", return_value="old"):
                with self.assertRaises(ValueError):
                    module.closed_startup("current")
            with patch.object(module, "owners", return_value=[{"fd": "other"}]):
                with self.assertRaises(ValueError):
                    module.closed_startup("current")
            with self.assertRaises(ValueError):
                module.closed_startup("stale")


if __name__ == "__main__":
    unittest.main()
