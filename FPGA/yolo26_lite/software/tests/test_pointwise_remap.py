"""用独立逐像素矩阵运算核验分组／逆排列，并拒绝不满足点卷积条件的图。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from remap_pointwise import transform, transform_matmul, parse_geometry


def value(identity, shape, layout="NCHWc16", parameter=False):
    result = dict(_type_key="icraft::xir::Params" if parameter else "icraft::xir::Value",
                  v_id=identity, mtype="@hostm", name="", tags={}, dtype=dict(
                      _type_key="icraft::xir::TensorType", shape=shape, layout="@layout(" + layout + ")",
                      element_dtype=dict(storage_dtype="@fp(16)", scale="@scales([axis=-1])"),
                      merged_distrs=["@merged_distr([1, 4]^ZERO:{v:[0, %d)})" % (shape[1] * 16)]))
    return result


def example():
    inp = value(1, [1, 2, 4, 4, 16])
    weights = value(2, [3, 2, 1, 1, 16, 16], "OIHWo16i16", True)
    bias = value(3, [3, 16], "Oo16", True)
    conv_out, final = value(4, [1, 3, 4, 4, 16]), value(5, [1, 3, 4, 4, 16])
    conv = dict(_type_key="icraft::xir::Conv2d", op_id=10, name="", inputs=[inp, weights, bias],
                outputs=[conv_out], groups=1, stride_height=1, stride_width=1, dilation_height=1,
                dilation_width=1, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                compile_target="@zhuget(330)", cut_scale="@scales([axis=-1])")
    activation = dict(_type_key="icraft::xir::SiLU", op_id=11, name="", inputs=[conv_out], outputs=[final])
    generated = dict(ops=[dict(op_id=100, name="icraft::xir::Conv2dNode", origin_ops_id=[10, 11])])
    return generated, dict(ops=[conv, activation])


def pointwise_oracle(inp, weights, bias):
    n, cb, h, w, lane = inp.shape
    dense = inp.transpose(0, 1, 4, 2, 3).reshape(n, cb * lane, h * w)
    result = np.einsum("oc,ncp->nop", weights, dense) + bias[None, :, None]
    result = result / (1 + np.exp(-result))
    return result.reshape(n, -1, lane, h, w).transpose(0, 1, 3, 4, 2)


def int8_example():
    generated, adapted = example()
    for op in adapted["ops"]:
        for val in op["inputs"] + op["outputs"]:
            dtype = val["dtype"]
            dtype["layout"] = dtype["layout"].replace("16", "32")
            dtype["shape"] = [32 if n == 16 else n for n in dtype["shape"]]
            element = dtype["element_dtype"]
            if len(dtype["shape"]) == 5:
                channels = dtype["shape"][1] * 32
                dtype["merged_distrs"] = ["@merged_distr([1, 4]^ZERO:{v:[0, %d)})" % channels]
                # 三个激活各有独立 scale，重排不能将它们互换或统一为输入 scale。
                scale = {1: 0.125, 4: 0.25, 5: 0.0625}[val["v_id"]]
                element.update(storage_dtype="@sint(8)", scale="@scales([axis=-1]%d*%s)" % (channels, scale),
                               zero_points=[])
            elif len(dtype["shape"]) == 6:
                element.update(storage_dtype="@sint(8)", scale="@scales([axis=4]96*0.03125)")
    return generated, adapted


class PointwiseRemapTest(unittest.TestCase):
    def test_int8_reorder_preserves_quantized_pixels_weights_and_scales(self):
        generated, adapted = int8_example()
        candidate, records = transform(generated, adapted, [100], 2)
        rng = np.random.RandomState(54)
        inp = rng.randint(-64, 64, (1, 2, 4, 4, 32)).astype(np.int8)
        weights = rng.randint(-3, 4, (96, 64)).astype(np.float32) * np.float32(0.03125)
        bias = rng.randint(-2, 3, 96).astype(np.float32)

        def quantized_oracle(pixels):
            dense = pointwise_oracle(pixels.astype(np.float32) * np.float32(0.125), weights, bias)
            return np.clip(np.rint(dense / np.float32(0.0625)), -128, 127).astype(np.int8)

        values = {1: inp}
        for op in candidate["ops"]:
            source = values[op["inputs"][0]["v_id"]]
            kind = op["_type_key"].split("::")[-1]
            if kind == "Reshape":
                result = source.reshape(op["shape"])
            elif kind == "Transpose":
                result = source.transpose(op["dims"])
            elif kind == "Conv2d":
                result = quantized_oracle(source)
                self.assertEqual(op["inputs"][1:], adapted["ops"][0]["inputs"][1:])
                self.assertEqual(op["cut_scale"], adapted["ops"][0]["cut_scale"])
            else:
                self.assertEqual(kind, "SiLU")
                result = source
            values[op["outputs"][0]["v_id"]] = result
        np.testing.assert_array_equal(values[5], quantized_oracle(inp))
        originals = [adapted["ops"][0]["inputs"][0], adapted["ops"][0]["outputs"][0],
                     adapted["ops"][1]["outputs"][0]]
        expected = [originals[0], originals[0], originals[1], originals[2], originals[2], originals[2]]
        for op, original in zip(candidate["ops"], expected):
            self.assertEqual(op["outputs"][0]["dtype"]["element_dtype"], original["dtype"]["element_dtype"])
        self.assertEqual(records[0]["reorder_logical_read_write_bytes"], 2 * (64 + 96) * 16)
        self.assertEqual(adapted, int8_example()[1])
        aspect, _ = transform(generated, adapted, [100], 2, mode="aspect")
        self.assertEqual(aspect["ops"][0]["outputs"][0]["dtype"]["shape"],
                         [1, 2, 8, 2, 32])

    def test_int8_rejects_unproven_scale_and_packed_encodings(self):
        for field, replacement in (("scale", "@scales([axis=1]64*0.125)"),
                                   ("scale", "@scales([axis=-1]32*0.125)"),
                                   ("scale", "@scales([axis=-1]64*-0.125)"),
                                   ("scale", "@scales([axis=-1]64*1e999)"),
                                   ("scale", "@scales([axis=-1]32*0.125,32*0.25)"),
                                   ("zero_points", [1]), ("storage_dtype", "@uint(8)")):
            generated, adapted = int8_example()
            adapted["ops"][0]["inputs"][0]["dtype"]["element_dtype"][field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                transform(generated, adapted, [100], 2)
        generated, adapted = int8_example()
        with self.assertRaisesRegex(ValueError, "Unsupported packed activation lane"):
            transform_matmul(generated, adapted, adapted, [100])

    def test_aspect_keeps_pixel_order_and_uses_no_permutation(self):
        generated, adapted = example()
        candidate, records = transform(generated, adapted, [100], 2, mode="aspect")
        self.assertEqual([op["_type_key"].split("::")[-1] for op in candidate["ops"]],
                         ["Reshape", "Conv2d", "SiLU", "Reshape"])
        self.assertEqual(candidate["ops"][0]["outputs"][0]["dtype"]["shape"], [1, 2, 8, 2, 16])
        self.assertEqual(candidate["ops"][2]["outputs"][0]["dtype"]["shape"], [1, 3, 8, 2, 16])
        rng = np.random.RandomState(41)
        inp = rng.randint(-2, 3, (1, 2, 4, 4, 16)).astype(np.float32)
        weights = rng.randint(-1, 2, (48, 32)).astype(np.float32)
        bias = rng.randint(-2, 3, 48).astype(np.float32)
        actual = pointwise_oracle(inp.reshape(1, 2, 8, 2, 16), weights, bias).reshape(1, 3, 4, 4, 16)
        np.testing.assert_array_equal(actual, pointwise_oracle(inp, weights, bias))
        self.assertEqual(records[0]["reorder_logical_read_write_bytes"], 0)
        for width in (0, 3, 4):
            with self.subTest(width=width), self.assertRaises(ValueError):
                transform(generated, adapted, [100], width, mode="aspect")

    def test_direct_output_requires_convolution_consumers_and_restored_shape(self):
        generated, adapted = example()
        consumer = deepcopy(adapted["ops"][0])
        consumer["op_id"] = 12
        consumer["inputs"][0] = deepcopy(adapted["ops"][1]["outputs"][0])
        adapted["ops"].append(consumer)
        candidate, records = transform(generated, adapted, [100], 2, True, "aspect", [100])
        self.assertFalse(records[0]["explicit_output_copy"])
        self.assertEqual(candidate["ops"][-1], consumer)
        self.assertEqual(candidate["ops"][-2]["outputs"], consumer["inputs"][:1])
        adapted["ops"].append(dict(_type_key="icraft::xir::Output", op_id=13,
                                   inputs=consumer["inputs"][:1], outputs=[]))
        with self.assertRaisesRegex(ValueError, "convolution readers or one contiguous Split"):
            transform(generated, adapted, [100], 2, True, "aspect", [100])
        with self.assertRaisesRegex(ValueError, "distinct selected"):
            transform(generated, adapted, [100], 2, True, "aspect", [999])

    def test_fanout_reads_the_same_restored_value_without_copy(self):
        generated, adapted = example()
        final = deepcopy(adapted["ops"][1]["outputs"][0])
        for identity in (20, 21, 22):
            adapted["ops"].append(dict(_type_key="icraft::xir::Conv2d", op_id=identity,
                inputs=[deepcopy(final)], outputs=[value(identity, [1, 3, 4, 4, 16])]))
        candidate, records = transform(generated, adapted, [100], 2, True, direct_outputs=[100])
        restored = candidate["ops"][-4]["outputs"][0]
        self.assertEqual(restored, final)
        self.assertTrue(all(op["inputs"][0] == restored for op in candidate["ops"][-3:]))
        self.assertFalse(records[0]["explicit_output_copy"])
        self.assertEqual(records[0]["direct_consumers"], [20, 21, 22])
        adapted["ops"][-1]["inputs"][0]["dtype"]["shape"][2] = 2
        with self.assertRaisesRegex(ValueError, "same restored tensor"):
            transform(generated, adapted, [100], 2, True, direct_outputs=[100])

    def test_direct_split_requires_all_channels_and_original_geometry(self):
        generated, adapted = example()
        split = dict(_type_key="icraft::xir::Split", op_id=20, axis=1, sections=[1, 2],
                     inputs=[deepcopy(adapted["ops"][1]["outputs"][0])],
                     outputs=[value(6, [1, 1, 4, 4, 16]), value(7, [1, 2, 4, 4, 16])])
        adapted["ops"].append(split)
        candidate, records = transform(generated, adapted, [100], 2, True, direct_outputs=[100])
        self.assertFalse(records[0]["explicit_output_copy"])
        self.assertEqual(candidate["ops"][-1], split)
        for field, replacement in (("axis", 2), ("sections", [1, 1])):
            invalid = deepcopy(adapted)
            invalid["ops"][-1][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                transform(generated, invalid, [100], 2, True, direct_outputs=[100])

    def test_per_operation_geometry_does_not_change_other_mapping(self):
        generated, adapted = example()
        second = deepcopy(adapted)
        for op in second["ops"]:
            op["op_id"] += 10
            for val in op["inputs"] + op["outputs"]:
                val["v_id"] += 10
        adapted["ops"].extend(second["ops"])
        generated["ops"].append(dict(op_id=200, name="icraft::xir::Conv2dNode", origin_ops_id=[20, 21]))
        candidate, records = transform(generated, adapted, [100, 200], 2,
            aspect_operations=[200], edge_overrides=[(100, 1)], width_overrides=[(200, 1)])
        self.assertEqual(records[0]["tiled_shape"], [16, 2, 1, 1, 16])
        self.assertEqual(records[1]["tiled_shape"], [1, 2, 16, 1, 16])
        self.assertEqual(records[0]["output_shape"], records[1]["output_shape"])
        for overrides in (dict(edge_overrides=[(200, 1)]), dict(width_overrides=[(100, 1)]),
                          dict(edge_overrides=[(100, 1), (100, 2)]), dict(edge_overrides=[(100, 0)])):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                transform(generated, adapted, [100, 200], 2, aspect_operations=[200], **overrides)
        self.assertEqual(parse_geometry("3240:10"), (3240, 10))

    def test_nontrivial_channels_and_pixels_match_independent_oracle(self):
        generated, adapted = example()
        candidate, records = transform(generated, adapted, [100], 2)
        rng = np.random.RandomState(19)
        inp = rng.randint(-2, 3, (1, 2, 4, 4, 16)).astype(np.float32)
        weights = rng.randint(-1, 2, (48, 32)).astype(np.float32)
        bias = rng.randint(-2, 3, 48).astype(np.float32)
        values = {1: inp}
        for op in candidate["ops"]:
            source = values[op["inputs"][0]["v_id"]]
            operation_kind = op["_type_key"].split("::")[-1]
            if operation_kind == "Reshape":
                result = source.reshape(op["shape"])
            elif operation_kind == "Transpose":
                result = source.transpose(op["dims"])
            elif operation_kind == "Conv2d":
                result = pointwise_oracle(source, weights, bias)
            else:
                # oracle 已含 SiLU，保持变换图的输出 ID 连通；数学仅计算一次。
                self.assertEqual(operation_kind, "SiLU")
                result = source
            values[op["outputs"][0]["v_id"]] = result
        np.testing.assert_array_equal(values[5], pointwise_oracle(inp, weights, bias))
        self.assertEqual(records[0]["internal_batch"], 4)
        self.assertEqual(records[0]["reorder_logical_read_write_bytes"], 2 * (32 + 48) * 16 * 2)
        self.assertEqual(adapted, example()[1])
        ids = [op["op_id"] for op in candidate["ops"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reject_geometry_groups_padding_and_partial_tile(self):
        for field, changed in (("stride_width", 2), ("pad_top", 1), ("groups", 2)):
            generated, adapted = example()
            adapted["ops"][0][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                transform(generated, adapted, [100], 2)
        for edge in (0, 3, 4):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                transform(*example(), [100], edge)

    def test_reject_other_consumers_and_padded_channels(self):
        generated, adapted = example()
        extra = deepcopy(adapted["ops"][1])
        extra["op_id"] = 12
        adapted["ops"].append(extra)
        with self.assertRaises(ValueError):
            transform(generated, adapted, [100], 2)
        generated, adapted = example()
        adapted["ops"][0]["inputs"][0]["dtype"]["merged_distrs"] = []
        with self.assertRaises(ValueError):
            transform(generated, adapted, [100], 2)

    def test_reject_non_pointwise_kernel_and_duplicate_selection(self):
        generated, adapted = example()
        adapted["ops"][0]["inputs"][1]["dtype"]["shape"][2] = 3
        with self.assertRaises(ValueError):
            transform(generated, adapted, [100], 2)
        with self.assertRaises(ValueError):
            transform(*example(), [100, 100], 2)

    def test_explicit_copy_preserves_public_output_identity(self):
        generated, adapted = example()
        candidate, records = transform(generated, adapted, [100], 2, copy_output=True)
        last = candidate["ops"][-1]
        self.assertEqual(last["_type_key"], "icraft::xir::Copy")
        self.assertEqual(last["outputs"], adapted["ops"][-1]["outputs"])
        self.assertNotEqual(last["inputs"][0]["v_id"], 5)
        self.assertEqual(records[0]["reorder_logical_read_write_bytes"], 2 * (32 + 2 * 48) * 16 * 2)

    def test_logical_matmul_preserves_weight_order_bias_and_activation(self):
        generated, adapted = example()
        logical = deepcopy(adapted)
        for op in logical["ops"]:
            for val in op["inputs"] + op["outputs"]:
                shape = val["dtype"]["shape"]
                if len(shape) == 5:
                    val["dtype"].update(shape=[1, 4, 4, shape[1] * 16], layout="@layout(NHWC)", merged_distrs=[])
                elif len(shape) == 6:
                    val["dtype"].update(shape=[1, 1, 32, 48], layout="@layout(HWIO)", merged_distrs=[])
        candidate, _ = transform_matmul(generated, adapted, logical, [100])
        matrix = candidate["ops"][1]
        self.assertEqual(matrix["inputs"][1]["dtype"]["shape"], [1, 32, 48])
        self.assertEqual(matrix["inputs"][1]["dtype"]["layout"], "@layout(*IO)")
        self.assertEqual(matrix["inputs"][1]["v_id"], 2)
        self.assertEqual(matrix["inputs"][2], logical["ops"][0]["inputs"][2])
        self.assertEqual(candidate["ops"][2]["_type_key"], "icraft::xir::SiLU")
        rng = np.random.RandomState(23)
        pixels = rng.randint(-2, 3, (1, 4, 4, 32)).astype(np.float32)
        weights = rng.randint(-2, 3, (1, 1, 32, 48)).astype(np.float32)
        direct = np.einsum("nhwc,co->nhwo", pixels, weights[0, 0])
        reshaped = (pixels.reshape(1, 16, 32) @ weights.reshape(1, 32, 48)).reshape(1, 4, 4, 48)
        np.testing.assert_array_equal(direct, reshaped)


if __name__ == "__main__":
    unittest.main()
