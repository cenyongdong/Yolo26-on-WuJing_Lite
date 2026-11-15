"""边界读取跨过层完成点时，不把整箱全部归给某个卷积。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_bus_timeline import FIELDS
from analyze_prefetch import region_observation, transition_bounds


class PrefetchComparisonTest(unittest.TestCase):
    def setUp(self):
        self.counters = dict(samples=[
            dict(begin_ns=10, end_ns=20, layer_before=26, layer_after=26),
            dict(begin_ns=30, end_ns=40, layer_before=26, layer_after=27),
            dict(begin_ns=80, end_ns=90, layer_before=27, layer_after=28)])
        self.bus = dict(timeline=[])
        for index, bounds in enumerate(((20, 25, 45, 50), (45, 50, 70, 75), (70, 75, 95, 100))):
            row = dict.fromkeys(FIELDS, index + 1)
            row.update(index=index, begin_host_ns_bounds=list(bounds[:2]), end_host_ns_bounds=list(bounds[2:]))
            self.bus["timeline"].append(row)

    def test_read_interval_and_whole_bin_bounds(self):
        self.assertEqual(transition_bounds(self.counters["samples"], 27), [30, 40])
        region = region_observation(self.bus, self.counters, 27, 1)
        self.assertEqual(region["start_host_ns_bounds"], [30, 40])
        self.assertEqual(region["end_host_ns_bounds"], [80, 90])
        self.assertEqual(region["wholly_inside_bins"], [1])
        self.assertEqual(region["possibly_intersecting_bins"], [0, 1, 2])
        self.assertEqual(region["event_count_bounds"]["r_accepted"], [2, 6])
        self.assertEqual(region["interface_return_payload_bytes_bounds"], [128, 384])
        self.assertIsNone(region["mpe_wait_ms"])
        self.assertIsNone(region["ddr_bytes"])

    def test_no_whole_bin_is_a_valid_zero_lower_bound(self):
        self.bus["timeline"] = self.bus["timeline"][::2]
        region = region_observation(self.bus, self.counters, 27, 1)
        self.assertEqual(region["wholly_inside_bins"], [])
        self.assertEqual(region["event_count_bounds"]["r_accepted"], [0, 4])

    def test_unobserved_completion_rejected(self):
        with self.assertRaises(ValueError):
            region_observation(self.bus, self.counters, 28, 1)
        with self.assertRaises(ValueError):
            transition_bounds(self.counters["samples"], 26)


if __name__ == "__main__":
    unittest.main()
