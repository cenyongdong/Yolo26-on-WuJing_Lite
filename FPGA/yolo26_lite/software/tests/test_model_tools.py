"""离线模型工具的输入表示和断点身份检查，不启动编译器或 SDK。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from compile_icraft import completed_stages, make_config, review_graph
from common import sha256
from compare_simulation import compare, read_dump, verify_input_pair
from simulate_model import encode_input


def graph_fixture(element_type="@fp(32)"):
    return dict(ai_target="@zhuget(330)", icraft_version="v3.39.0", params_bytes=4,
                params_md5=hashlib.md5(b"test").hexdigest(), ops=[
                    dict(_type_key="icraft::xir::Input", outputs=[dict(dtype=dict(
                        shape=[1, 640, 640, 3], layout="@layout(NHWC)", element_dtype=element_type))]),
                    dict(_type_key="icraft::xir::Output", inputs=[dict(dtype=dict(
                        shape=[1, 84, 8400], layout="@layout(**C)", element_dtype="@fp(32)"))])])


class InputRepresentationTest(unittest.TestCase):
    def test_rgb_bytes_and_fp32_have_identical_meaning(self):
        rgb = np.arange(3 * 640 * 640, dtype=np.uint32).astype(np.uint8).reshape(1, 3, 640, 640)
        image = rgb.astype(np.float32) / np.float32(255)
        byte_input, dtype = encode_input(image, graph_fixture("@uint(8)"))
        float_input, _ = encode_input(image, graph_fixture())
        self.assertEqual(dtype, "@uint(8)")
        self.assertEqual(byte_input.nbytes, 1228800)
        self.assertEqual(float_input.nbytes, 4915200)
        np.testing.assert_array_equal(byte_input, rgb.transpose(0, 2, 3, 1))
        np.testing.assert_array_equal(byte_input.astype(np.float32) / np.float32(255), float_input)

    def test_uint8_rejects_rounding_or_invalid_inputs(self):
        graph = graph_fixture("@uint(8)")
        for value in (-0.0, 0.5, -0.1, 1.1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_input(np.full((1, 3, 640, 640), value, dtype=np.float32), graph)
        with self.assertRaises(ValueError):
            encode_input(np.zeros((1, 3, 640, 640), dtype=np.float64), graph)

    def test_graph_input_contract_is_explicit(self):
        image = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for key, value in (("layout", "@layout(NCHW)"), ("shape", [1, 640, 640, 4]),
                           ("element_dtype", "@fp(16)")):
            graph = graph_fixture()
            graph["ops"][0]["outputs"][0]["dtype"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode_input(image, graph)

    def test_config_retains_baseline_and_scales_once(self):
        # 保护数值边界；文本换行不作为模型语义的失败条件。
        baseline = make_config()
        self.assertIn('inputs_dtype = "FP32"', baseline)
        self.assertIn('pre_scale = "nop"', baseline)
        candidate = make_config("rgb_u8")
        self.assertIn('inputs_dtype = "UInt8"', candidate)
        self.assertIn('pre_scale = [255.0, 255.0, 255.0]', candidate)
        self.assertIn('pre_method = "nop"', candidate)
        with self.assertRaises(ValueError):
            make_config("uint8_with_second_normalization")


class CompilerResumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.identity = dict(onnx_sha256="a" * 64, config_sha256="b" * 64,
                             runtime_manifest_sha256="c" * 64, input_profile="fp32")
        self.save("input-identity.json", self.identity)
        self.save("yolo26l_parsed.json", graph_fixture())
        (self.directory / "yolo26l_parsed.raw").write_bytes(b"test")
        self.save("parse.review.json", review_graph(self.directory / "yolo26l_parsed.json"))
        self.history = [dict(stage="parse", run=dict(native_exit=0))]
        self.save("progress.json", self.history)

    def save(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def test_resumes_only_unchanged_successful_prefix(self):
        self.assertEqual(completed_stages(self.directory, self.identity), self.history)
        for field in self.identity:
            changed = dict(self.identity, **{field: "changed"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                completed_stages(self.directory, changed)
        for record in (dict(stage="optimize", run=dict(native_exit=0)),
                       dict(stage="parse", run=dict(native_exit=1))):
            self.save("progress.json", [record])
            with self.assertRaises(ValueError):
                completed_stages(self.directory, self.identity)

    def test_rejects_changed_graph_or_raw(self):
        graph = graph_fixture()
        graph["unexpected_change"] = True
        self.save("yolo26l_parsed.json", graph)
        with self.assertRaises(ValueError):
            completed_stages(self.directory, self.identity)
        self.save("yolo26l_parsed.json", graph_fixture())
        (self.directory / "yolo26l_parsed.raw").write_bytes(b"fail")
        with self.assertRaises(ValueError):
            completed_stages(self.directory, self.identity)


class SimulationPairingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input_path = self.directory / "case-2-input.npy"
        rgb = np.arange(3 * 640 * 640, dtype=np.uint32).astype(np.uint8).reshape(1, 3, 640, 640)
        image = rgb.astype(np.float32) / np.float32(255)
        np.save(str(self.input_path), image)
        self.host = self.directory / "host"
        self.generated = self.directory / "generated"
        self.records = {}
        for directory, stage, element_type in ((self.host, "quantized", "@fp(32)"),
                                                (self.generated, "ZG", "@uint(8)")):
            dump_directory = directory / "native/model/ftmpSFB"
            dump_directory.mkdir(parents=True)
            np.zeros((1, 84, 8400), dtype="<f4").tofile(str(dump_directory / "7.ftmp"))
            encoded, _ = encode_input(image, graph_fixture(element_type))
            encoded.tofile(str(directory / "input.ftmp"))
            record = dict(run=dict(native_exit=0), case=2, stage=stage, output_value_ids=[7],
                          input_sha256=sha256(self.input_path), graph_sha256="a" * 64,
                          input_element_type=element_type, encoded_input_bytes=encoded.nbytes,
                          encoded_input_sha256=sha256(directory / "input.ftmp"))
            self.records[directory] = record
            self.save_record(directory, record)

    def save_record(self, directory, record):
        (directory / "simulation.json").write_text(json.dumps(record), encoding="utf-8")

    def verify(self):
        return verify_input_pair(self.input_path, self.host, self.generated)

    def test_matches_input_meaning_across_representations_and_legacy_record(self):
        result = self.verify()
        self.assertEqual(result["case"], 2)
        self.assertEqual(result["host"]["encoded_input_bytes"], 4915200)
        self.assertEqual(result["generated"]["encoded_input_bytes"], 1228800)
        # 旧 FP32 模拟没有编码摘要；仍逐字节从原始输入重建并核对，不能直接跳过校验。
        legacy = dict(self.records[self.host])
        for field in ("input_element_type", "encoded_input_bytes", "encoded_input_sha256"):
            del legacy[field]
        self.save_record(self.host, legacy)
        self.assertEqual(self.verify(), result)

    def test_rejects_mixed_cases_input_identity_stage_and_failed_native_run(self):
        original = self.records[self.generated]
        for field, value in (("case", 1), ("input_sha256", "b" * 64), ("stage", "quantized"),
                             ("run", dict(native_exit=1)), ("output_value_ids", [7, 8])):
            self.save_record(self.generated, dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()
        self.save_record(self.generated, original)
        self.verify()

    def test_rejects_changed_encoded_content_even_with_updated_record_hash(self):
        path = self.generated / "input.ftmp"
        original = path.read_bytes()
        for content in (original[:-1], bytes([original[0] ^ 1]) + original[1:]):
            path.write_bytes(content)
            record = dict(self.records[self.generated], encoded_input_bytes=len(content),
                          encoded_input_sha256=sha256(path))
            self.save_record(self.generated, record)
            with self.subTest(size=len(content)), self.assertRaises(ValueError):
                self.verify()

    def test_dump_requires_exact_bytes_success_and_valid_reference_tensor(self):
        actual, path = read_dump(self.host)
        self.assertEqual(actual.shape, (1, 84, 8400))
        # np.fromfile 会忽略不满一个 float 的尾字节；必须先检查完整文件长度。
        with path.open("ab") as stream:
            stream.write(b"x")
        with self.assertRaises(ValueError):
            read_dump(self.host)
        self.save_record(self.generated, dict(self.records[self.generated], run=dict(native_exit=9)))
        with self.assertRaises(ValueError):
            read_dump(self.generated)
        for invalid in (actual.reshape(84, 8400), actual.astype(np.float64), actual + np.float32(np.nan)):
            with self.assertRaises(ValueError):
                compare(invalid, actual)


if __name__ == "__main__":
    unittest.main()
