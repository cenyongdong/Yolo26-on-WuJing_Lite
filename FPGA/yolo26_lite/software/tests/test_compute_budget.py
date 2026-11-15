"""核验 grouped convolution 和广播矩阵乘法计数，不把逻辑运算数冒充物理执行量。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_compute_budget import operation_count, analyze


def value(shape, layout, precision="@sint(8)"):
    return dict(dtype=dict(shape=shape, layout="@layout(" + layout + ")", element_dtype=precision))


class ComputeBudgetTest(unittest.TestCase):
    def test_grouped_kernel_counts_each_product_once(self):
        op = dict(_type_key="icraft::xir::Conv2d", op_id=1, groups=2,
                  inputs=[value([1, 6, 6, 4], "NHWC"), value([3, 3, 2, 6], "HWIO")],
                  outputs=[value([1, 4, 4, 6], "NHWC")])
        # 16 个输出像素 × 6 通道；每点 3×3×2 个乘加。
        self.assertEqual(operation_count(op), ("@sint(8)", 16 * 6 * 18 * 2))
        report = analyze(dict(ops=[op]), int8_tops=8)
        self.assertAlmostEqual(report["ideal_conv_matmul_ms"], 3456 / 8e9)
        op["groups"] = 1
        with self.assertRaises(ValueError):
            operation_count(op)

    def test_broadcast_matrices_and_mismatched_output(self):
        op = dict(_type_key="icraft::xir::Matmul", op_id=2,
                  inputs=[value([2, 1, 3, 4], "***C", "@fp(16)"),
                          value([1, 5, 4, 6], "***C", "@fp(16)")],
                  outputs=[value([2, 5, 3, 6], "***C", "@fp(16)")])
        self.assertEqual(operation_count(op), ("@fp(16)", 2 * 5 * 3 * 4 * 6 * 2))
        op["outputs"][0]["dtype"]["shape"] = [2, 1, 3, 6]
        with self.assertRaises(ValueError):
            operation_count(op)

    def test_ignored_ops_and_unknown_peak_are_explicit(self):
        self.assertIsNone(operation_count(dict(_type_key="icraft::xir::Softmax")))
        report = analyze(dict(ops=[dict(_type_key="icraft::xir::Softmax")]))
        self.assertEqual(report["excluded_operator_counts"], {"Softmax": 1})
        self.assertIsNone(report["ideal_conv_matmul_fps"])
        with self.assertRaises(ValueError):
            analyze(dict(ops=[]), int8_tops=float("nan"))


if __name__ == "__main__":
    unittest.main()
