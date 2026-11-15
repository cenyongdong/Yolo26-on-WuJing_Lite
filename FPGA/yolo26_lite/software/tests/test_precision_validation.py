"""代理精度评价验证坐标还原、类别匹配和内容隔离；不改变无 NMS 输出。"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import save_json, sha256
from validate_precision import checked_dataset, evaluate, original_boxes


class PrecisionValidationTest(unittest.TestCase):
    def test_actual_xy_scales_padding_clipping_and_input_ownership(self):
        detections = np.array([[9, 18, 109, 78, .9, 5]], dtype=np.float32)
        original = detections.copy()
        restored = original_boxes(detections, dict(original=[40, 20], padding=[10, 20], scale_xy=[2, 3]))
        np.testing.assert_allclose(restored[0, :4], [0, 0, 40, 58 / 3], rtol=1e-6)
        np.testing.assert_array_equal(detections, original)
        with self.assertRaises(ValueError):
            original_boxes(detections * np.nan, dict(original=[40, 20], padding=[10, 20], scale_xy=[2, 3]))

    def test_correct_class_has_ap_wrong_class_does_not(self):
        dense = np.full((1, 84, 8400), -20, dtype=np.float32)
        dense[:, :4] = 1
        dense[0, 9, 0] = 10
        row = dict(case=0, labels_yolo=[[5, 6 / 640, 6 / 640, 12 / 640, 12 / 640]],
                   transform=dict(original=[640, 640], padding=[0, 0], scale_xy=[1, 1]))
        with tempfile.TemporaryDirectory() as tmp:
            correct = evaluate([row], [dense], Path(tmp) / "config")
            self.assertGreater(correct["map50_95"], .99)
            self.assertTrue(all(row["map"] > .99 for row in correct["map_by_iou"]))
            self.assertFalse(correct["nms"])
            tighter_box = copy.deepcopy(row)
            tighter_box["labels_yolo"][0][3] = 10 / 640
            imperfect = evaluate([tighter_box], [dense], Path(tmp) / "config")
            self.assertGreater(imperfect["map50"], .99)
            self.assertEqual(imperfect["map_by_iou"][-1], dict(iou=.95, map=0))
            self.assertLess(imperfect["map50_95"], imperfect["map50"])
            wrong = copy.deepcopy(row)
            wrong["labels_yolo"][0][0] = 6
            incorrect = evaluate([wrong], [dense], Path(tmp) / "config")
            self.assertEqual(incorrect["map50_95"], 0)
            with self.assertRaises(ValueError):
                evaluate([row], [], Path(tmp) / "config")

    def test_validation_rejects_calibration_overlap_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "validation").mkdir()
            image = path / "validation" / "image.npy"
            image.write_bytes(b"fixed-input")
            manifest = dict(content_disjoint=True, labelled_proxy_only=True,
                            calibration=[dict(sha256="calibration")],
                            validation=[dict(sha256="held-out", input="image.npy", input_sha256=sha256(image))])
            save_json(path / "manifest.json", manifest)
            self.assertEqual(len(checked_dataset(path)[1]), 1)
            manifest["validation"][0]["sha256"] = "calibration"
            save_json(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                checked_dataset(path)
            manifest["validation"][0]["sha256"] = "held-out"
            save_json(path / "manifest.json", manifest)
            image.write_bytes(b"changed-input")
            with self.assertRaisesRegex(ValueError, "changed"):
                checked_dataset(path)


if __name__ == "__main__":
    unittest.main()
