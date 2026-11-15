"""混合策略保护敏感路径，并验证原生成功后可恢复审查而不重复执行。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import quantize_mixed as module
from common import save_json, sha256


def fixture_graph(complete_psa=False):
    operations = []

    def append(kind, inputs=(), channels=64, kernel=1):
        identifier = len(operations)
        values = [copy.deepcopy(operations[value]["outputs"][0]) for value in inputs]
        if kind == "Conv2d":
            values.append(dict(v_id=10000 + identifier, dtype=dict(shape=[kernel, kernel, 64, channels])))
        operations.append(dict(op_id=identifier, _type_key="icraft::xir::" + kind, inputs=values,
                               outputs=[dict(v_id=identifier, dtype=dict(shape=[1, 20, 20, channels]))]))
        return identifier

    image = append("Input")
    stem = append("SiLU", [append("Conv2d", [image], kernel=3)])
    backbone = append("Conv2d", [stem], kernel=3)
    attention_outputs = []
    for _ in range(3):
        boundary = append("Conv2d", [backbone], kernel=3) if complete_psa else backbone
        attention_input = append("Conv2d", [boundary], channels=128) if complete_psa else boundary
        first = append("Matmul", [append("Transpose", [attention_input])])
        softmax = append("Softmax", [append("Multiply", [first])])
        last = append("Transpose", [append("Matmul", [softmax, attention_input])])
        if complete_psa:
            position = append("Conv2d", [attention_input], kernel=3)
            projected = append("Conv2d", [append("Add", [last, position])])
            residual = append("Add", [boundary, projected])
            hidden = append("SiLU", [append("Conv2d", [residual], channels=128)], channels=128)
            attention_outputs.append(append("Add", [residual, append("Conv2d", [hidden])]))
        else:
            attention_outputs.append(append("Conv2d", [last], kernel=3))
    heads = [append("Conv2d", [attention_outputs[index % 3]], channels=4 if index < 3 else 80)
             for index in range(6)]
    tail = append("Concat", [append("Reshape", [head]) for head in heads])
    append("Output", [append("Transpose", [tail])])
    return dict(ops=operations)


class MixedPrecisionTest(unittest.TestCase):
    def test_native_csv_covers_ports_and_preserves_sensitive_paths(self):
        graph = fixture_graph()
        rows = module.precision_policy(graph, "head_attention")
        self.assertEqual(len(rows), len(graph["ops"]))
        self.assertEqual(len({row["op_id"] for row in rows}), len(rows))
        for op, row in zip(graph["ops"], rows):
            kind = module.kind(op)
            self.assertEqual(row["type"], op["_type_key"] + "Node")
            if kind != "Conv2d":
                self.assertEqual(row["dtype"], "fp16")
            elif op["op_id"] == 1 or op["outputs"][0]["dtype"]["shape"][-1] in (4, 80):
                self.assertEqual(row["dtype"], "fp16")
            else:
                self.assertEqual(row["dtype"], "int8")

    def test_changed_head_or_attention_contract_is_rejected(self):
        graph = fixture_graph()
        for target in ("Softmax", "Concat"):
            changed = copy.deepcopy(graph)
            op = next(op for op in changed["ops"] if module.kind(op) == target)
            op["_type_key"] = "icraft::xir::Add"
            with self.subTest(target=target), self.assertRaises(ValueError):
                module.precision_policy(changed, "head_attention")

    def test_full_detection_head_stops_at_shared_neck_feature(self):
        graph = fixture_graph()
        head = next(op for op in graph["ops"] if module.kind(op) == "Conv2d" and
                    op["outputs"][0]["dtype"]["shape"][-1] == 4)
        shared = head["inputs"][0]["v_id"]
        branch = copy.deepcopy(head)
        branch["op_id"] = 999
        branch["outputs"][0].update(v_id=999, dtype=dict(shape=[1, 20, 20, 64]))
        branch["inputs"][1]["dtype"]["shape"] = [3, 3, 64, 64]
        head["inputs"][0]["v_id"] = 999
        graph["ops"].append(branch)
        base = {row["op_id"]: row["dtype"] for row in module.precision_policy(graph, "head_attention")}
        full = {row["op_id"]: row["dtype"] for row in module.precision_policy(graph, "detection_attention")}
        self.assertEqual(base[999], "int8")
        self.assertEqual(full[999], "fp16")
        self.assertEqual(full[shared], "int8")

    def test_complete_psa_protects_residual_region_but_not_its_producer(self):
        graph = fixture_graph(complete_psa=True)
        policy = {row["op_id"]: row["dtype"] for row in module.precision_policy(graph, "psa_detection")}
        self.assertEqual(policy[3], "int8")
        floating_convolutions = [op for op in graph["ops"] if module.kind(op) == "Conv2d" and
                                 policy[op["op_id"]] == "fp16"]
        self.assertEqual(len(floating_convolutions), 1 + 6 + 3 * 5)
        by_value = {op["outputs"][0]["v_id"]: op for op in graph["ops"]}
        tail = next(op for op in graph["ops"] if module.kind(op) == "Add" and
                    module.kind(by_value[op["inputs"][0]["v_id"]]) == "Add")
        tail["_type_key"] = "icraft::xir::Concat"
        with self.assertRaisesRegex(ValueError, "FFN residual"):
            module.precision_policy(graph, "psa_detection")

    def test_early_convolution_ablation_changes_only_the_second_stem(self):
        graph = fixture_graph()
        baseline = module.precision_policy(graph, "detection_attention")
        candidate = module.precision_policy(graph, "stem2_detection")
        changed = [new["op_id"] for old, new in zip(baseline, candidate) if old["dtype"] != new["dtype"]]
        # 本微型图的第二个卷积后没有 SiLU；真实 YOLO 图还有紧邻的 SiLU，一并保持浮点。
        self.assertEqual(changed, [3])

    def test_binary_forward_mode_and_partition_count(self):
        config = module.make_config(Path("calibration"), 96, "head_attention", native_batch=4)
        self.assertIn('forward_mode = "image"', config)
        self.assertIn('batch = 4', config)
        self.assertIn('qdtype = "int8"', config)
        self.assertIn('mix_precision = "./precision.csv"', config)
        self.assertNotIn('[parse]', config)
        with self.assertRaises(ValueError):
            module.make_config(Path("calibration"), 32, "head_attention", native_batch=3)

    def test_statistics_cannot_cross_weights_data_precision_or_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            donor = Path(tmp)
            logs = donor / "logs" / "quantize" / "yolo26l"
            logs.mkdir(parents=True)
            source = dict(json_sha256="graph", raw_sha256="weights")
            policy = [dict(op_id=0, type="InputNode", dtype="fp32")]
            save_json(donor / "input-identity.json", dict(source=source, calibration_manifest_sha256="data",
                      calibration_count=32, native_batch=1, precision_policy=policy))
            save_json(donor / "quantize.exit.json", dict(native_exit=0))
            for name in ("ftmps", "raws"):
                (logs / ("yolo26l_quantize_" + name + ".csv")).write_text("fixed statistics\n")
            result = module.checked_statistics(donor, source, "data", 32, 1, policy)
            self.assertEqual(set(result["files"]), {"ftmps", "raws"})
            for changed_source, data, count, batch, changed_policy in (
                    (dict(source, raw_sha256="new"), "data", 32, 1, policy),
                    (source, "changed-data", 32, 1, policy), (source, "data", 4, 1, policy),
                    (source, "data", 32, 4, policy), (source, "data", 32, 1, [])):
                with self.assertRaises(ValueError):
                    module.checked_statistics(donor, changed_source, data, count, batch, changed_policy)
            config = module.make_config(Path("calibration"), 32, "head_attention", 1, True)
            self.assertIn('ftmp_csv = "./calibration_ftmps.csv"', config)
            self.assertIn('raw_csv = "./calibration_raws.csv"', config)
            changed = [dict(policy[0], dtype="fp16")]
            features = module.checked_statistics(donor, source, "data", 32, 1, changed, "features")
            self.assertEqual(set(features["files"]), {"ftmps"})
            self.assertEqual(features["changed_precision_ids"], [0])
            config = module.make_config(Path("calibration"), 32, "head_attention", 1, "features")
            self.assertNotIn('raw_csv', config)

    def test_native_success_recovers_review_without_reexecuting(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calibration = output / "calibration"
            calibration.mkdir()
            save_json(calibration / "manifest.json", dict(target=[512, 384], lists={"0": dict(path="list.txt")},
                                                           calibration=[]))
            (calibration / "list.txt").write_text("")
            for name in ("quantize.toml", "precision.csv", "yolo26l_optimized.json", "yolo26l_optimized.raw"):
                (output / name).write_text("fixture")
            (output / "yolo26l_quantized.json").write_text(json.dumps(dict(ops=[])))
            source = {name + "_sha256": sha256(output / ("yolo26l_optimized." + name)) for name in ("json", "raw")}
            save_json(output / "input-identity.json", dict(source=source, calibration_root=str(calibration),
                      calibration_manifest_sha256=sha256(calibration / "manifest.json"), calibration_count=0,
                      calibration_list_sha256=sha256(calibration / "list.txt"),
                      config_sha256=sha256(output / "quantize.toml"), precision_csv_sha256=sha256(output / "precision.csv")))
            save_json(output / "quantize.exit.json", dict(native_exit=0))
            with patch.object(module, "compiler_environment", return_value={}), \
                    patch.object(module, "review_graph", return_value=dict(host_compute=[])) as review, \
                    patch.object(module, "run_logged") as run:
                module.execute(output, "quantize", output, output / "runtime.json")
                review.assert_called_once_with(output / "yolo26l_quantized.json", module.ModelGeometry(512, 384))
                run.assert_not_called()
                self.assertTrue((output / "quantize.review.json").exists())
                save_json(output / "quantize.exit.json", dict(native_exit=21013))
                with self.assertRaisesRegex(ValueError, "Preserve failed"):
                    module.execute(output, "quantize", output, output / "runtime.json")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
