"""同帧证据要验证完整像素和身份，单字节差异也必须失败。"""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_live_video import audit_sample


class LiveVideoAudit(unittest.TestCase):
    def test_padded_yuyv_one_box_and_corrupted_display(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            width, height, stride = 640, 480, 1288
            padded_row = bytes([128] * (width * 2) + [0xFF] * (stride - width * 2))
            source = padded_row * (height - 1) + padded_row[:width * 2]
            (run / "native-reference-source.yuyv").write_bytes(source)
            (run / "native-reference-rgb.bin").write_bytes(bytes([130]) * (512 * 384 * 3))
            score = 1 / (1 + math.exp(-2))
            tensor = np.zeros((1, 84, 4032), dtype="<f4")
            tensor[0, 4:] = -20
            tensor[0, 11, 0] = 2
            tensor[0, :4, 0] = [-3.5, -4.5, 11.5, 12.5]
            tensor.tofile(run / "native-reference-output.f32")
            frame = dict(epoch=26, frame=3, captured_ns=100, inference_completed_ns=200, adopted_ns=300,
                         adopted_hardware_tag=1, adopted_address=0x4000, adopted_status=0x142, adopted_faults=0,
                         width=width, height=height, stride=stride, bytes_used=len(source),
                         model_width=512, model_height=384, model_stride=1536, model_bytes=512 * 384 * 3,
                         resized_width=512, resized_height=384, left=0, top=0, gain=0.8,
                         model_boxes=[[32, 40, 96, 104, score, 7]], source_boxes=[[40, 50, 120, 130, score, 7]])
            (run / "native-reference-frame.json").write_text(json.dumps(frame))
            timing = dict(epoch=26, frame=3, captured_ns=100, completed_ns=200)
            times_path = run / "native-live-times.json"
            times_path.write_text(json.dumps([timing]))
            display = np.zeros((720, 1280), dtype="<u2")
            display[:, 160:1120] = 0x8410
            display[75:195, 220:340] = 0xFD20
            display[77:193, 222:338] = 0x8410
            display.tofile(run / "native-reference-composed.rgb565")
            scan_path = run / "native-reference-scan-buffer.rgb565"
            display.tofile(scan_path)
            summary = dict(format=dict(width=width, height=height, stride=stride),
                           conversion=dict(matrix=0), adopted_reference_boxes=1)
            audit, _ = audit_sample(run, summary)
            self.assertEqual((audit["boxes"], audit["tensor_bytes"], audit["display_rgb565_bytes"]),
                             (1, 1354752, 1843200))
            with scan_path.open("r+b") as damaged:
                damaged.seek((75 * 1280 + 220) * 2)
                damaged.write(b"\x00")
            with self.assertRaisesRegex(AssertionError, "native-reference-scan-buffer"):
                audit_sample(run, summary)
            timing["frame"] = 4
            times_path.write_text(json.dumps([timing]))
            with self.assertRaisesRegex(ValueError, "Same-frame timing"):
                audit_sample(run, summary)


if __name__ == "__main__":
    unittest.main()
