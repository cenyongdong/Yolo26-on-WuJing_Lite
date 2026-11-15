"""仅用合成记录测试解析边界，测试数据不充当板端观测。"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_bus_timeline import analyze


class BusTimelineTest(unittest.TestCase):
    def setUp(self):
        self.trace = dict(schema=1, clock_period_ns_from_constraints=5, period_cycles=100, capacity=512,
                          status=2 << 16 | 2, arm_begin_ns=10000, arm_end_ns=10020,
                          stop_begin_ns=10980, stop_end_ns=11020,
                          records=[[0, 100, 1000, 0, 4, 8, 20, 2, 5, 1, 300, 5, 0, 50, 10, 20],
                                   [1, 100, 1100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0]])

    def test_totals_and_no_unobserved_bandwidth_claim(self):
        report = analyze(self.trace)
        self.assertEqual(report["totals"]["ar_accepted"], 4)
        self.assertEqual(report["interface_return_payload_bytes"], 1280)
        self.assertEqual(report["interface_return_payload_MBps"], 1280)
        self.assertEqual(report["timeline"][1]["begin_host_ns_bounds"], [10500, 10520])
        self.assertEqual(report["timeline"][1]["no_valid_cycle_fraction"], 1)
        self.assertIsNone(report["mpe_wait_ms"])
        self.assertIsNone(report["ddr_bandwidth_bytes_per_second"])

    def test_corruption_rejected(self):
        for field, value in ((0, 5), (1, 0), (2, 1120), (4, 101), (10, 1), (14, 1 << 27)):
            data = copy.deepcopy(self.trace)
            data["records"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze(data)

    def test_partial_last_and_visible_overflow(self):
        self.trace["records"][1][1] = 20
        self.trace["records"][1][13] = 20
        self.trace["status"] |= 4
        self.assertTrue(analyze(self.trace)["overflow"])
        self.trace["records"][0][1] = 90
        with self.assertRaises(ValueError):
            analyze(self.trace)

    def test_nonfrozen_and_stale_records_rejected(self):
        for status in (1, 2, 2 << 16 | 3, 2 << 16 | 10):
            with self.subTest(status=status), self.assertRaises(ValueError):
                analyze(dict(self.trace, status=status))


if __name__ == "__main__":
    unittest.main()
