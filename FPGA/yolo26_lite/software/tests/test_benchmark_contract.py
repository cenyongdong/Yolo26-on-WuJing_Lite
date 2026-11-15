"""CPU 回执只接受有限的真实测量；纯 Host 测试不导入板端控制器。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from benchmark_contract import REQUIRED_MODULES, validate_cpu_receipt


class BenchmarkContractTest(unittest.TestCase):
    def fixture(self):
        rows = [dict(kind="cpu_module", module=name, iterations=100, mean_ns=12, p50_ns=10,
                     p95_ns=15, p99_ns=17, maximum_ns=20, npu_executed=False) for name in REQUIRED_MODULES]
        return rows + [dict(kind="summary", sdk_starts=0, camera_starts=0, npu_executed=False)]

    def test_baseline_and_new_measurement_are_both_valid(self):
        rows = self.fixture()
        self.assertEqual(validate_cpu_receipt(rows, 100), REQUIRED_MODULES)
        rows.insert(0, dict(rows[0], module="rgb888_to_nhwc_u8"))
        self.assertIn("rgb888_to_nhwc_u8", validate_cpu_receipt(rows, 100))

    def test_missing_duplicate_or_wrong_budget_rejected(self):
        for rows in ([], self.fixture()[1:], self.fixture()[:-1], self.fixture()[:-1] + self.fixture()):
            with self.assertRaises(ValueError):
                validate_cpu_receipt(rows, 100)
        with self.assertRaises(ValueError):
            validate_cpu_receipt(self.fixture(), 101)

    def test_scope_and_invalid_timing_cannot_pass(self):
        for change in (dict(npu_executed=True), dict(p95_ns=9), dict(maximum_ns=float("nan")),
                       dict(mean_ns=-1), dict(iterations=True)):
            rows = self.fixture()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cpu_receipt(rows, 100)
        for key, value in (("sdk_starts", 1), ("camera_starts", False), ("npu_executed", 0)):
            rows = self.fixture()
            rows[-1][key] = value
            with self.assertRaises(ValueError):
                validate_cpu_receipt(rows, 100)


if __name__ == "__main__":
    unittest.main()
