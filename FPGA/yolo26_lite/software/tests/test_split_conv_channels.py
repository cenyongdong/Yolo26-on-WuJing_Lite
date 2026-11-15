"""验证分块的量化轴、完整输出覆盖及拒绝边界；完整模型数值另由原厂生成代码模拟。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from split_conv_channels import slice_scale, transform


def value(identity, shape, layout, merged, scale, storage="@sint(8)", parameter=False):
    return dict(_type_key="icraft::xir::Params" if parameter else "icraft::xir::Value",
                v_id=identity, mtype="@hostm", name="", tags={}, dtype=dict(
                    shape=shape, layout="@layout(" + layout + ")", merged_distrs=merged,
                    element_dtype=dict(expressed_dtype="@fp(32)", storage_dtype=storage,
                                       scale=scale, zero_points=[])))


def fixture():
    source = value(0, [1, 1, 8, 8, 32], "NCHWc32", ["@merged_distr([1, 4]^ZERO:{v:[0, 32)})"],
                   "@scales([axis=-1]32*0.02)")
    middle = value(3, [1, 4, 4, 4, 32], "NCHWc32", ["@merged_distr([1, 4]^ZERO:{v:[0, 128)})"],
                   "@scales([axis=-1]128*0.03)")
    final = deepcopy(middle)
    final["v_id"] = 4
    final["dtype"]["element_dtype"]["scale"] = "@scales([axis=-1]128*0.04)"
    weights = value(1, [4, 1, 3, 3, 32, 32], "OIHWo32i32",
                    ["@merged_distr([1, 5]^ZERO:{v:[0, 32)})", "@merged_distr([0, 4]^ZERO:{v:[0, 128)})"],
                    "@scales([axis=4]" + ",".join(str((i + 1) / 1000) for i in range(128)) + ")",
                    parameter=True)
    bias = value(2, [4, 32], "Oo32", ["@merged_distr([0, 1]^ZERO:{v:[0, 128)})"],
                 "@scales([axis=-1])", storage="@tf(32)", parameter=True)
    conv = dict(_type_key="icraft::xir::Conv2d", op_id=1, inputs=[source, weights, bias], outputs=[middle],
                groups=1, stride_height=2, stride_width=2, dilation_height=1, dilation_width=1)
    silu = dict(_type_key="icraft::xir::SiLU", op_id=2, inputs=[deepcopy(middle)], outputs=[final])
    generated = dict(ops=[dict(_type_key="icraft::xir::HardOp", name="icraft::xir::Conv2dNode",
                               op_id=3405, origin_ops_id=[1, 2])])
    return generated, dict(ops=[conv, silu])


class ChannelSplitTests(unittest.TestCase):
    def test_complete_ordered_output_and_reduction_preserved(self):
        generated, adapted = fixture()
        original = deepcopy(adapted)
        graph, record = transform(generated, adapted, 3405, 64)
        self.assertEqual(adapted, original)
        self.assertEqual(graph["ops"][-1]["outputs"], original["ops"][-1]["outputs"])
        self.assertEqual(graph["ops"][-1]["axis"], 1)
        self.assertEqual([(r["begin_channel"], r["end_channel"]) for r in record["branches"]], [(0, 64), (64, 128)])
        for op in graph["ops"][::2][:-1]:
            self.assertEqual(op["inputs"][0], original["ops"][0]["inputs"][0])
            self.assertEqual(op["inputs"][1]["dtype"]["shape"], [2, 1, 3, 3, 32, 32])
            self.assertEqual(op["outputs"][0]["dtype"]["shape"], [1, 2, 4, 4, 32])

    def test_scale_tokens_remain_exact(self):
        self.assertEqual(slice_scale("@scales([axis=4]1e-09,0.0033447195,4.50,7)", 1, 3, 4, 4),
                         "@scales([axis=4]0.0033447195,4.50)")
        self.assertEqual(slice_scale("@scales([axis=-1]256*0.04044174)", 32, 96, 256, 4),
                         "@scales([axis=-1]64*0.04044174)")
        for scale in ("@scales([axis=3]1,2)", "@scales([axis=4]1)", "@scales([axis=-1]3*0.5)"):
            with self.assertRaises(ValueError):
                slice_scale(scale, 0, 1, 2, 4)

    def test_reject_partial_blocks_and_wrong_kernel(self):
        generated, adapted = fixture()
        for block in (0, 16, 96, 128, True):
            with self.assertRaises(ValueError):
                transform(generated, adapted, 3405, block)
        adapted["ops"][0]["inputs"][1]["dtype"]["shape"][2] = 1
        with self.assertRaises(ValueError):
            transform(generated, adapted, 3405, 64)

    def test_reject_extra_reader_and_asymmetric_input(self):
        generated, adapted = fixture()
        adapted["ops"].append(dict(op_id=3, inputs=[deepcopy(adapted["ops"][0]["outputs"][0])], outputs=[]))
        with self.assertRaises(ValueError):
            transform(generated, adapted, 3405, 64)
        generated, adapted = fixture()
        adapted["ops"][0]["inputs"][0]["dtype"]["element_dtype"]["zero_points"] = [1]
        with self.assertRaises(ValueError):
            transform(generated, adapted, 3405, 64)


if __name__ == "__main__":
    unittest.main()
