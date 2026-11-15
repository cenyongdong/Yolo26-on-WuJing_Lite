"""防止把分区变化归因为数据数量，或把不同留出子集当成精度改进。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from compare_precision_ablation import check_factor, check_protocol, load_experiment
from common import save_json, sha256


class PrecisionAblationTest(unittest.TestCase):
    def setUp(self):
        self.identity = dict(source=dict(json_sha256="graph", raw_sha256="weights"),
                             calibration_manifest_sha256="data", calibration_count=32, native_batch=1,
                             precision_policy=[dict(op_id=3, type="Conv2d", dtype="int8")])

    def test_partition_must_not_masquerade_as_more_calibration_data(self):
        expanded = dict(self.identity, calibration_count=96, native_batch=6)
        with self.assertRaisesRegex(ValueError, "images per partition"):
            check_factor(self.identity, expanded, "calibration_count")
        grouped = dict(self.identity, native_batch=2)
        check_factor(self.identity, grouped, "partition")
        check_factor(grouped, expanded, "calibration_count")

    def test_precision_changes_keep_measurement_and_model_fixed(self):
        changed = deepcopy(self.identity)
        changed["precision_policy"][0]["dtype"] = "fp16"
        check_factor(self.identity, changed, "precision")
        changed["calibration_count"] = 96
        with self.assertRaisesRegex(ValueError, "changed calibration"):
            check_factor(self.identity, changed, "precision")
        changed = deepcopy(self.identity)
        changed["source"]["raw_sha256"] = "new-weights"
        with self.assertRaisesRegex(ValueError, "weights"):
            check_factor(self.identity, changed, "precision")

    def test_partial_inputs_or_different_score_threshold_are_rejected(self):
        metric = dict(images=32, labels=218, classes=46, nms=False, max_det=300, confidence_filter=.001,
                      metrics_library="fixed", map50=.7, map50_95=.6,
                      identity=dict(dataset_manifest_sha256="data", cases=list(range(32)), full_held_out_set=True))
        check_protocol(metric, deepcopy(metric))
        for change in ("cases", "partial", "threshold"):
            changed = deepcopy(metric)
            if change == "cases":
                changed["identity"]["cases"][-1] = 33
            elif change == "partial":
                changed["identity"]["full_held_out_set"] = False
            else:
                changed["confidence_filter"] = .25
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_protocol(metric, changed)

    def test_complete_outputs_are_bound_to_the_original_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model, results, dataset = [directory / name for name in ("model", "results", "dataset")]
            for path in (model, results, dataset):
                path.mkdir()
            for suffix in ("json", "raw"):
                (model / ("yolo26l_ZG." + suffix)).write_bytes(b"fixed-model")
            manifest_path = dataset / "manifest.json"
            save_json(manifest_path, dict(validation=[dict(case=0, input_sha256="input")]))
            model_identity = dict(calibration_root=str(dataset), calibration_manifest_sha256=sha256(manifest_path))
            validation_identity = dict(graph_sha256=sha256(model / "yolo26l_ZG.json"),
                                       raw_sha256=sha256(model / "yolo26l_ZG.raw"),
                                       dataset_manifest_sha256=sha256(manifest_path), cases=[0])
            save_json(model / "input-identity.json", model_identity)
            save_json(model / "generate.review.json", dict(
                graph=dict(json_sha256=validation_identity["graph_sha256"],
                           raw_sha256=validation_identity["raw_sha256"], host_compute=[]),
                run=dict(native_exit=0)))
            save_json(results / "identity.json", validation_identity)
            save_json(results / "metrics.json", dict(identity=validation_identity, images=1))
            output = results / "case-0-dense.npy"
            output.write_bytes(b"complete-output")
            save_json(results / "case-0-dense.npy.json", dict(input_sha256="input", output_sha256=sha256(output)))
            load_experiment(model, results)
            output.write_bytes(b"partial-output")
            with self.assertRaisesRegex(ValueError, "numerical output changed"):
                load_experiment(model, results)
            output.write_bytes(b"complete-output")
            manifest = json.loads(manifest_path.read_text())
            manifest["validation"][0]["input_sha256"] = "different-input"
            save_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                load_experiment(model, results)


if __name__ == "__main__":
    unittest.main()
