"""存储前缀约束与卷积计算量的独立例子；不把候选条件误记为可部署证明。"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_layout import convolution_rows, copy_groups, parse_loops


def value(identity, count, location):
    return dict(v_id=identity, dtype=dict(shape=[count], element_dtype="@fp(16)"), mtype=location)


def operation(identity, kind, inputs, outputs, layers=1):
    return dict(op_id=identity, _type_key="icraft::xir::HardOp", name="icraft::xir::" + kind + "Node",
                inputs=inputs, outputs=outputs, layer_count=layers)


def copy_example():
    source, left, right = value(1, 8, "@chunk(1@0)"), value(2, 4, "@chunk(1@0)"), value(3, 4, "@chunk(1@8)")
    a, b, c = value(4, 4, "@chunk(2@0)"), value(5, 4, "@chunk(2@8)"), value(6, 4, "@chunk(2@16)")
    combined = value(7, 12, "@chunk(2@0)")
    return dict(ops=[operation(1, "Input", [], [source], 0), operation(2, "Split", [source], [left, right], 0),
                     operation(3, "Copy", [left], [a]), operation(4, "Copy", [right], [b]),
                     operation(5, "Conv2d", [right], [c]), operation(6, "Concat", [a, b, c], [combined], 0)])


class LayoutAnalysisTest(unittest.TestCase):
    def test_prefix_requires_all_slices_and_no_other_prefix_writer(self):
        graph = copy_example()
        row = copy_groups(graph, {3: 0.1, 4: 0.2})[0]
        self.assertTrue(row["candidate"])
        self.assertEqual((row["prefix_bytes"], row["logical_copy_io_bytes"]), (16, 32))
        self.assertAlmostEqual(row["mean_copy_ms"], 0.3)
        self.assertFalse(row["producer_direct_write_proven"])
        graph["ops"].append(operation(7, "Write", [], [value(8, 1, "@chunk(2@4)")]))
        self.assertIn("destination_prefix_has_other_writers", copy_groups(graph, {})[0]["exclusion_reasons"])
        graph = copy_example()
        graph["ops"][3]["inputs"] = [graph["ops"][1]["outputs"][0]]
        reasons = copy_groups(graph, {})[0]["exclusion_reasons"]
        self.assertIn("not_all_split_outputs_copied", reasons)
        self.assertIn("copy_does_not_preserve_prefix_layout", reasons)

    def test_nonzero_offset_and_real_concat_are_not_prefix_candidates(self):
        graph = copy_example()
        graph["ops"][-1]["layer_count"] = 1
        graph["ops"][-1]["outputs"][0]["mtype"] = "@chunk(2@8)"
        reasons = copy_groups(graph, {})[0]["exclusion_reasons"]
        self.assertIn("nonzero_parent_offset", reasons)
        self.assertIn("split_or_concat_has_instructions", reasons)

    def test_loop_parser_keeps_bounds_and_rejects_ambiguous_origins(self):
        text = ("[I] PerformLoopTransform, op_name = icraft::xir::Conv2dNode, op_ofm_vid = 8, nested_loop =\n"
                " for kt_MPE_t, ct_MPE_t, in 4, 2 : # PARALLEL, REDUCTION\n")
        self.assertEqual(parse_loops(text)[("Conv2dNode", 8)], [{"kt_MPE_t": 4, "ct_MPE_t": 2}])
        with self.assertRaises(ValueError):
            parse_loops(text + text)
        with self.assertRaises(ValueError):
            parse_loops(text.replace("4, 2 :", "4 :"))

    def test_group_convolution_flops_and_missing_log_remain_explicit(self):
        def tensor(identity, shape, layout):
            return dict(v_id=identity, dtype=dict(shape=shape, layout="@layout(" + layout + ")",
                                                  element_dtype="@fp(16)"))
        inp = tensor(1, [1, 4, 4, 8], "NHWC")
        weight = tensor(2, [3, 3, 4, 12], "HWIO")
        out = tensor(3, [1, 2, 2, 12], "NHWC")
        logical = dict(op_id=1, _type_key="icraft::xir::Conv2d", inputs=[inp, weight], outputs=[out], groups=2)
        adapted = copy.deepcopy(logical)
        adapted["op_id"] = 9
        generated = operation(10, "Conv2d", [inp], [out])
        generated.update(origin_ops_id=[9], params=[weight])
        rows = convolution_rows(dict(ops=[generated]), dict(ops=[adapted]), dict(ops=[logical]), {}, {10: 2.0})
        self.assertEqual(rows[0]["flops"], 2 * 4 * 12 * 3 * 3 * 4)
        self.assertIsNone(rows[0]["examine_input_reuse"])
        self.assertFalse(rows[0]["loop_log_available"])
        self.assertEqual(rows[0]["effective_gflops"], rows[0]["flops"] / 2e6)


if __name__ == "__main__":
    unittest.main()
