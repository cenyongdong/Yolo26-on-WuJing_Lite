"""校准数据必须隔离，RGB／布局／归一化必须与部署契约一致。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from prepare_calibration import partition, prepare_pixels, read_labels


class CalibrationTest(unittest.TestCase):
    def test_content_split_is_disjoint_order_independent_and_rejects_duplicates(self):
        rows = [dict(image=str(i), sha256=str(i)) for i in range(8)]
        first, second = partition(rows, 6, "fixed")
        self.assertEqual((first, second), partition(list(reversed(rows)), 6, "fixed"))
        self.assertFalse({row["sha256"] for row in first} & {row["sha256"] for row in second})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            partition(rows + [dict(rows[0], image="duplicate")], 6, "fixed")
        for count in (0, 8, 9):
            with self.assertRaises(ValueError):
                partition(rows, count, "fixed")

    def test_color_padding_and_exact_normalized_representation(self):
        image = np.full((2, 4, 3), [10, 20, 200], dtype=np.uint8)
        rgb, normalized, transform = prepare_pixels(image, target=8)
        self.assertEqual(transform["resized"], [8, 4])
        self.assertEqual(transform["padding"], [0, 2])
        np.testing.assert_array_equal(rgb[2:6], np.broadcast_to([200, 20, 10], (4, 8, 3)))
        self.assertTrue(np.all(rgb[:2] == 114) and np.all(rgb[6:] == 114))
        np.testing.assert_array_equal(normalized.view(np.uint32), (rgb.astype(np.float32) / np.float32(255)).view(np.uint32))
        self.assertEqual(normalized.dtype, np.float32)

    def test_labels_reject_illegal_class_and_nonfinite_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            self.assertEqual(read_labels(path), ([], False))
            path.write_text("0 .5 .5 .2 .4\n", encoding="utf-8")
            self.assertEqual(read_labels(path), ([[0, .5, .5, .2, .4]], True))
            for row in ("80 .5 .5 .2 .4", "0 nan .5 .2 .4", "0 .5 .5 0 .4", "0.5 .5 .5 .2 .4"):
                path.write_text(row, encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    read_labels(path)


if __name__ == "__main__":
    unittest.main()
