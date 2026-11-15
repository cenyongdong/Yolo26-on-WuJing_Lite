"""计时归属和对照身份的独立小例；不伪造实际硬件计时结果。"""
import copy
from pathlib import Path
import sys
import struct
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_npu import (compare_runs, inspect_timer, layer_durations, map_layers, statistics_ms,
                         summarize_placement, tensor_bytes, verify_timer_encoding)


def op(identity, count):
    return dict(_type_key="icraft::xir::HardOp", op_id=identity, layer_count=count, name="icraft::xir::TestNode",
                origin_ops_id=[identity + 100], inputs=[], outputs=[], instr=[])


class NpuAnalysisTest(unittest.TestCase):
    def test_actual_order_zero_layers_and_multilayer_operations(self):
        graph = dict(ops=[op(7, 0), op(2, 2), op(9, 1)])
        initialized = dict(completion_layers=3, fusion=[dict(first_layer=0, layer_count=3, operation=99,
                                                            originals=[9, 7, 2])])
        rows = map_layers(graph, initialized)
        self.assertEqual([(r["operation"], r["first_layer"], r["layer_count"]) for r in rows],
                         [(9, 0, 1), (7, 1, 0), (2, 1, 2)])
        for members in ([9, 7, 7], [9, 2], [9, 7, 55]):
            invalid = copy.deepcopy(initialized)
            invalid["fusion"][0]["originals"] = members
            with self.assertRaises(ValueError):
                map_layers(graph, invalid)
        initialized["fusion"][0]["layer_count"] = 4
        with self.assertRaises(ValueError):
            map_layers(graph, initialized)

    def test_timer_interpretation_is_explicit(self):
        sample = dict(expected_layers=3, returned_layer_count=3, api_layer_ms=[1.0, 3.0, 4.0])
        self.assertIsNone(layer_durations(sample, 3, "unverified"))
        self.assertEqual(layer_durations(sample, 3, "durations"), [1, 3, 4])
        self.assertEqual(layer_durations(sample, 3, "cumulative"), [1, 2, 1])
        for values in ([1, 2], [1, float("nan"), 3], [1, -1, 2], [1, None, 2]):
            with self.assertRaises(ValueError):
                layer_durations(dict(sample, api_layer_ms=values), 3, "durations")
        with self.assertRaises(ValueError):
            layer_durations(dict(sample, api_layer_ms=[2, 1, 3]), 3, "cumulative")

    def test_storage_dtype_and_aliases_are_not_peak_memory(self):
        value = dict(dtype=dict(shape=[1, 4, 160, 160, 16], element_dtype=dict(storage_dtype="@fp(16)")))
        self.assertEqual(tensor_bytes(value), 3276800)
        root = dict(value_id=10, is_ocm=True, is_host=False, real=True)
        alias = dict(root, value_id=11, real=False)
        deployment = dict(backends=[dict(backend_id=0, selected_ocm_strategy=2, merge_hardops=True,
                                        etm_reuse=True, values=[root, alias], segments=[])])
        summary = summarize_placement(deployment)[0]
        self.assertEqual((summary["ocm_values"], summary["ocm_real_values"], summary["ocm_alias_values"]), (2, 1, 1))
        self.assertNotIn("peak_memory_bytes", summary)

    def test_invalid_native_timer_keeps_wall_clock_comparison_separate(self):
        sample = dict(frame=15, expected_layers=3, returned_layer_count=3, api_layer_ms=[1.0, -3.0, 4.0],
                      api_calc_time_ms=2.0, collection_ns=123, read_tile_counts=[4], write_tile_counts=[5])
        check, durations = inspect_timer(sample, 3, "unverified")
        self.assertIsNotNone(check["validation_error"])
        self.assertIsNone(durations)
        self.assertEqual(check["api_sum_ms"], 2.0)
        sample["api_layer_ms"][1] = None
        check, _ = inspect_timer(sample, 3, "unverified")
        self.assertIsNone(check["api_sum_ms"])
        with self.assertRaises(ValueError):
            inspect_timer(sample, 3, "durations")

    def test_raw_timer_encoding_is_checked_without_fitting_a_scale(self):
        sample = dict(expected_layers=3, returned_layer_count=3, api_layer_ms=[0.001024, 0.002048, 0.001024],
                      api_calc_time_ms=0.004096,
                      timer_storage=[dict(captured=True, allocation_bytes=12,
                                          encoding="hex_in_increasing_byte_offset",
                                          bytes_hex=struct.pack("<3I", 4, 12, 16).hex())])
        result = verify_timer_encoding(sample, 1_000_000_000, 3)
        self.assertEqual(result["counter_last"], 16)
        self.assertLess(result["raw_api_max_error_ms"], 1e-12)
        with self.assertRaises(ValueError):
            verify_timer_encoding(sample, 500_000_000, 3)
        sample["timer_storage"][0]["bytes_hex"] = struct.pack("<3I", 4, 3, 16).hex()
        with self.assertRaises(ValueError):
            verify_timer_encoding(sample, 1_000_000_000, 3)

    def test_comparison_rejects_confounded_inputs_and_reused_boot(self):
        report = dict(identities={"model": "a", "input": "b"}, boot_sha256="c", boot_id="first", strategy=1,
                      sdk_libraries={"xrt": "sdk"},
                      placement=[dict(selected=1)], output_sha256="out", kinds_ms={},
                      timings=dict(forward_wait_ms={}, complete_call_ms={}))
        other = dict(report, boot_id="second", strategy=2, placement=[dict(selected=2)])
        self.assertTrue(compare_runs([report, other])["all_complete_outputs_bitwise_equal"])
        for change in (dict(identities={"input": "different"}), dict(boot_id="first"), dict(strategy=1),
                       dict(sdk_libraries={"xrt": "different"})):
            with self.assertRaises(ValueError):
                compare_runs([report, dict(other, **change)])
        self.assertFalse(compare_runs([report, dict(other, output_sha256="different")])["all_complete_outputs_bitwise_equal"])
        self.assertEqual(statistics_ms([1, 2, 3])["p99"], 3)


if __name__ == "__main__":
    unittest.main()
