"""独立逐像素参考核验真实二维块、整数上采样、拼接顺序和恢复分支。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from propagate_pointwise import propagate
from test_pointwise_remap import pointwise_oracle, value


def convolution(identity, input_value, channels, output_id):
    shape = list(input_value["dtype"]["shape"])
    shape[1] = channels
    pre, final = value(output_id, shape), value(output_id + 1, shape)
    conv = dict(_type_key="icraft::xir::Conv2d", op_id=identity, name="", groups=1,
        inputs=[input_value, value(output_id + 2, [channels, input_value["dtype"]["shape"][1], 1, 1, 16, 16],
                                  "OIHWo16i16", True), value(output_id + 3, [channels, 16], "Oo16", True)],
        outputs=[pre], stride_height=1, stride_width=1, dilation_height=1, dilation_width=1,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, cut_scale="@scales([axis=-1])")
    return conv, dict(_type_key="icraft::xir::SiLU", op_id=identity + 1, name="", inputs=[pre], outputs=[final])


def example(reverse=False):
    high_input, low_input = value(1, [1, 2, 8, 12, 16]), value(2, [1, 2, 4, 6, 16])
    high, low = convolution(10, high_input, 3, 100), convolution(20, low_input, 2, 200)
    high_final, low_final = high[1]["outputs"][0], low[1]["outputs"][0]
    up = value(210, [1, 2, 8, 12, 16])
    joined = value(220, [1, 5, 8, 12, 16])
    upsample = dict(_type_key="icraft::xir::Upsample", op_id=22, inputs=[low_final], outputs=[up],
                    height=8, width=12, interpolation="NEAREST", align_corners=False)
    concat = dict(_type_key="icraft::xir::Concat", op_id=23, axis=1,
                  inputs=[high_final, up] if reverse else [up, high_final], outputs=[joined])
    target = convolution(30, joined, 3, 300)
    split = dict(_type_key="icraft::xir::Split", op_id=32, axis=1, sections=[1, 2],
                 inputs=target[1]["outputs"], outputs=[value(310, [1, 1, 8, 12, 16]), value(311, [1, 2, 8, 12, 16])])
    # 高分辨率的额外卷积分支仍读取完整空间张量；测试只检查其输入，不改写该算子。
    high_reader = dict(_type_key="icraft::xir::Conv2d", op_id=12, inputs=[high_final], outputs=[])
    ops = [dict(_type_key="icraft::xir::Input", op_id=0, inputs=[], outputs=[high_input, low_input]),
           *high, high_reader, *low, upsample, concat, *target, split]
    generated = dict(ops=[dict(op_id=identity * 10, name="icraft::xir::Conv2dNode",
                              origin_ops_id=[identity, identity + 1]) for identity in (10, 20, 30)])
    return generated, dict(ops=ops)


def execute(graph, initial):
    values = dict(initial)
    for op in graph["ops"]:
        kind = op["_type_key"].split("::")[-1]
        if kind == "Input" or not op["outputs"]:
            continue
        operands = [values[v["v_id"]] for v in op["inputs"] if v["_type_key"] != "icraft::xir::Params"]
        source = operands[0]
        if kind == "Conv2d":
            channels = op["outputs"][0]["dtype"]["shape"][1] * 16
            weights = np.zeros((channels, source.shape[1] * 16), np.float32)
            weights[np.arange(channels), (np.arange(channels) * 7) % weights.shape[1]] = 1
            result = pointwise_oracle(source, weights, np.zeros(channels, np.float32))
        elif kind == "Reshape":
            result = source.reshape(op["shape"])
        elif kind == "Transpose":
            result = source.transpose(op["dims"])
        elif kind == "Upsample":
            result = source.repeat(op["height"] // source.shape[2], axis=2)
            result = result.repeat(op["width"] // source.shape[3], axis=3)
        elif kind == "Concat":
            result = np.concatenate(operands, axis=op["axis"])
        elif kind == "Split":
            parts = np.split(source, np.cumsum(op["sections"])[:-1], axis=op["axis"])
            values.update({out["v_id"]: part for out, part in zip(op["outputs"], parts)})
            continue
        else:
            # 逐像素参考已经包含 SiLU；Copy 则保持完整的数值排列。
            assert kind in ("SiLU", "Copy"), kind
            result = source.copy()
        values[op["outputs"][0]["v_id"]] = result
    return values


class LayoutPropagationTest(unittest.TestCase):
    def test_full_coordinates_channels_concat_order_and_restore(self):
        inputs = {1: (np.arange(1 * 2 * 8 * 12 * 16).reshape(1, 2, 8, 12, 16) % 101).astype(np.float32) / 32,
                  2: (np.arange(1 * 2 * 4 * 6 * 16).reshape(1, 2, 4, 6, 16) % 79).astype(np.float32) / 16}
        for reverse in (False, True):
            generated, original = example(reverse)
            saved = deepcopy(original)
            for edge in (1, 2):
                with self.subTest(reverse=reverse, edge=edge):
                    candidate, record = propagate(generated, original, 100, 200, 300, edge)
                    actual, expected = execute(candidate, inputs), execute(original, inputs)
                    for identity in record["output_values"]:
                        np.testing.assert_array_equal(actual[identity], expected[identity])
                    for identity in (10, 20, 30):
                        before = next(op for op in original["ops"] if op["op_id"] == identity)
                        after = next(op for op in candidate["ops"] if op["op_id"] == identity)
                        self.assertEqual(before["inputs"][1:], after["inputs"][1:])
                    self.assertEqual(len({op["op_id"] for op in candidate["ops"]}), len(candidate["ops"]))
            self.assertEqual(original, saved)

    def test_rejects_interpolation_geometry_and_unproved_fanout(self):
        for change in ("interpolation", "height", "fanout", "high_reader"):
            generated, original = example()
            if change in ("interpolation", "height"):
                up = next(op for op in original["ops"] if op["op_id"] == 22)
                up[change] = "BILINEAR" if change == "interpolation" else 7
            elif change == "fanout":
                original["ops"].append(dict(_type_key="icraft::xir::Copy", op_id=99,
                    inputs=[next(op for op in original["ops"] if op["op_id"] == 22)["outputs"][0]], outputs=[]))
            else:
                next(op for op in original["ops"] if op["op_id"] == 12)["_type_key"] = "icraft::xir::Upsample"
            with self.subTest(change=change), self.assertRaises(ValueError):
                propagate(generated, original, 100, 200, 300, 2)

    def test_rejects_partial_blocks_and_duplicate_selection(self):
        for edge in (0, 3, 4, -1):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                propagate(*example(), 100, 200, 300, edge)
        with self.assertRaisesRegex(ValueError, "distinct"):
            propagate(*example(), 100, 100, 300, 2)


if __name__ == "__main__":
    unittest.main()
