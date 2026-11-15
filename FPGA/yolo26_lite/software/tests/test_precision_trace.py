"""中间特征探针只能读取图中真实输出，不能接受未知或重复标识。"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from simulate_model import trace_values
from compare_precision_trace import logical_nhwc, tensor_metrics


class PrecisionTraceTest(unittest.TestCase):
    def test_trace_preserves_shape_layout_and_order(self):
        first = dict(v_id=4, dtype=dict(shape=[1, 4, 2, 2, 16], layout="NCHWc"))
        second = dict(v_id=7, dtype=dict(shape=[1, 2, 2, 64], layout="NHWC"))
        graph = dict(ops=[dict(outputs=[first]), dict(outputs=[second])])
        self.assertEqual(trace_values(graph, "7,4"), [second, first])
        self.assertEqual(trace_values(graph, ""), [])
        for identifiers in ("7,7", "-1", "8", "x"):
            with self.subTest(identifiers=identifiers), self.assertRaises(ValueError):
                trace_values(graph, identifiers)

    def test_logical_shape_and_metrics_are_not_a_second_physical_transpose(self):
        self.assertEqual(logical_nhwc(dict(dtype=dict(shape=[1, 4, 2, 3, 16], layout="@layout(NCHWc16)"))),
                         [1, 2, 3, 64])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            logical_nhwc(dict(dtype=dict(shape=[1, 2, 3, 64], layout="@layout(unknown)")))
        result = tensor_metrics(np.array([1, 2], dtype=np.float32), np.array([1, 1], dtype=np.float32))
        self.assertAlmostEqual(result["rms"], np.sqrt(.5))
        self.assertAlmostEqual(result["relative_l2"], np.sqrt(.2))
        self.assertAlmostEqual(result["cosine"], 3 / np.sqrt(10))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            tensor_metrics(np.array([1, np.nan]), np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
