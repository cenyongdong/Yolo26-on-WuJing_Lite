"""用人工寄存器观测验证边界和语义；合成数据不冒充板测。"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_counter_timeline import analyze


class CounterTimelineTest(unittest.TestCase):
    def fixture(self):
        return dict(schema=1, clock="host_steady_clock_ns", column_ids=[1, 2], truncated=False,
                    register_reads_are_atomic=False, ddr_bytes_measured=False, mpe_stall_cycles_measured=False,
                    samples=[dict(begin_ns=10, end_ns=20, layer_before=0, layer_after=0,
                                  reads=[0, 0], writes=[0, 0]),
                             dict(begin_ns=100, end_ns=120, layer_before=1, layer_after=2,
                                  reads=[0, 0], writes=[0, 0]),
                             dict(begin_ns=200, end_ns=210, layer_before=3, layer_after=3,
                                  reads=[0, 0], writes=[0, 0])])

    def test_zero_counters_are_not_idle_or_bandwidth(self):
        result = analyze(self.fixture(), 3)
        self.assertFalse(result["tile_activity_observed"])
        self.assertIsNone(result["ddr_bandwidth_bytes_per_second"])
        self.assertIsNone(result["mpe_wait_ms"])
        self.assertEqual(result["changes"][0]["layer_to"], 2)

    def test_counter_wrap_does_not_invent_bytes(self):
        trace = self.fixture()
        trace["samples"][0]["reads"][1] = 2 ** 32 - 2
        trace["samples"][1]["reads"][1] = 3
        trace["samples"][2]["reads"][1] = 3
        result = analyze(trace, 3)
        self.assertTrue(result["tile_activity_observed"])
        self.assertEqual(result["changes"][0]["read_delta_mod_uint32"][1], 5)
        self.assertEqual(result["counters"][2]["decreases"], 1)

    def test_nonatomic_sampling_and_truncation_are_retained(self):
        trace = self.fixture()
        trace["truncated"] = True
        result = analyze(trace, 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["changes"][0]["begin_ns"], 10)
        self.assertEqual(result["changes"][0]["end_ns"], 120)

    def test_reject_invalid_time_count_scope_and_claim(self):
        for change in (dict(begin_ns=5), dict(end_ns=99), dict(layer_before=3),
                       dict(reads=[0]), dict(writes=[0, -1])):
            trace = self.fixture()
            trace["samples"][1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(trace, 3)
        for change in (dict(column_ids=[0, 1]), dict(samples=[]), dict(ddr_bytes_measured=True)):
            trace = copy.deepcopy(self.fixture())
            trace.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(trace, 3)


if __name__ == "__main__":
    unittest.main()
