"""独立复核一张实测视频帧的 PL 像素、one-to-one 框及 HDMI 帧缓冲。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from color_reference import convert_q12
from common import save_json, sha256
from model_geometry import ModelGeometry
from probe_contract import require, validate_live_completion
from reference import decode_dense
from resize_reference import resize_fixed


DISPLAY_SIZE = (1280, 720)
MODEL = ModelGeometry(512, 384)
PALETTE = (0x07e0, 0xf800, 0x001f, 0xffe0, 0x07ff, 0xf81f, 0xffff, 0xfd20)
STAGES = ("capture", "preprocess", "inference", "display")


def letterbox(width, height, target_width, target_height):
    gain = min(target_width / width, target_height / height)
    resized_width, resized_height = round(width * gain), round(height * gain)
    return resized_width, resized_height, (target_width - resized_width) // 2, (target_height - resized_height) // 2, gain


def reference_boxes(tensor, threshold):
    boxes, scores = decode_dense(tensor, MODEL)
    logits = tensor[0, 4:].T
    # 按 logit 保留同分候选的索引次序，避免不同 sigmoid 实现舍入后改变同分排序。
    anchors = np.argsort(-logits.max(axis=1), kind="stable")[:300]
    selected = np.argsort(-logits[anchors].reshape(-1), kind="stable")[:300]
    locations, classes = anchors[selected // 80], selected % 80
    values = scores[locations, classes]
    rows = np.column_stack((boxes[locations], values, classes)).astype(np.float32)
    return rows[values >= np.float32(threshold)]


def restore_boxes(boxes, width, height, left, top, gain):
    rows = boxes.copy()
    rows[:, (0, 2)] = np.clip((rows[:, (0, 2)] - np.float32(left)) / np.float32(gain), 0, width)
    rows[:, (1, 3)] = np.clip((rows[:, (1, 3)] - np.float32(top)) / np.float32(gain), 0, height)
    return rows[(rows[:, 0] < rows[:, 2]) & (rows[:, 1] < rows[:, 3])]


def compose_rgb565(rgb, boxes):
    height, width, _ = rgb.shape
    display_width, display_height = DISPLAY_SIZE
    rw, rh, left, top, gain = letterbox(width, height, *DISPLAY_SIZE)
    x = (2 * np.arange(rw) + 1) * width // (2 * rw)
    y = (2 * np.arange(rh) + 1) * height // (2 * rh)
    red, green, blue = rgb[y[:, None], x].astype(np.uint16).transpose(2, 0, 1)
    frame = np.zeros((display_height, display_width), dtype="<u2")
    frame[top:top + rh, left:left + rw] = ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)
    for box in boxes:
        sx, sy, ex, ey = box[:4] * np.float32(gain)
        x1 = int(np.clip(np.floor(sx) + left, 0, display_width - 1))
        y1 = int(np.clip(np.floor(sy) + top, 0, display_height - 1))
        x2 = int(np.clip(np.ceil(ex) + left - 1, x1, display_width - 1))
        y2 = int(np.clip(np.ceil(ey) + top - 1, y1, display_height - 1))
        color = PALETTE[int(box[5]) % len(PALETTE)]
        for inset in range(2):
            a, b, c, d = x1 + inset, y1 + inset, x2 - inset, y2 - inset
            if a <= c and b <= d:
                frame[[b, d], a:c + 1] = color
                frame[b:d + 1, [a, c]] = color
    return frame


def audit_sample(run, summary):
    frame = json.loads((run / "native-reference-frame.json").read_text())
    times = json.loads((run / "native-live-times.json").read_text())
    matching = [value for value in times if (value["epoch"], value["frame"]) == (frame["epoch"], frame["frame"])]
    require(len(matching) == 1 and matching[0]["captured_ns"] == frame["captured_ns"] and
            matching[0]["completed_ns"] == frame["inference_completed_ns"] and
            frame["captured_ns"] <= frame["inference_completed_ns"] <= frame["adopted_ns"],
            "Same-frame timing does not match inference record")
    status = frame["adopted_status"]
    require(frame["adopted_hardware_tag"] > 0 and frame["adopted_address"] > 0 and
            frame["adopted_address"] % 64 == 0 and frame["adopted_faults"] == 0 and
            status & 0x142 == 0x142 and status & 0x80000081 == 0, "Display did not adopt an active locked buffer")

    width, height, stride = (summary["format"][name] for name in ("width", "height", "stride"))
    require((width, height, stride) == (frame["width"], frame["height"], frame["stride"]) and
            width > 0 and width % 2 == 0 and height > 0 and stride >= width * 2,
            "Invalid sampled YUYV geometry")
    source_path = run / "native-reference-source.yuyv"
    source = source_path.read_bytes()
    require(len(source) == frame["bytes_used"] and len(source) >= (height - 1) * stride + width * 2,
            "Incomplete original YUYV frame")
    yuyv = np.ndarray((height, width, 2), dtype=np.uint8, buffer=source, strides=(stride, 2, 1))
    matrix = summary["conversion"]["matrix"]
    require(matrix in range(4), "Unrecognized Q12 matrix")
    rgb = convert_q12(yuyv, matrix).reshape(height, width, 3)
    rw, rh, left, top, gain = letterbox(width, height, MODEL.width, MODEL.height)
    require((frame["model_width"], frame["model_height"], frame["model_stride"], frame["model_bytes"]) ==
            (MODEL.width, MODEL.height, MODEL.width * 3, MODEL.rgb_bytes) and
            (frame["resized_width"], frame["resized_height"], frame["left"], frame["top"], np.float32(frame["gain"])) ==
            (rw, rh, left, top, np.float32(gain)), "Sample transform differs from the rectangular input")
    expected_rgb = np.full((MODEL.height, MODEL.width, 3), 114, dtype=np.uint8)
    expected_rgb[top:top + rh, left:left + rw], _, _ = resize_fixed(rgb, rw, rh)
    rgb_path = run / "native-reference-rgb.bin"
    np.testing.assert_array_equal(np.fromfile(rgb_path, np.uint8), expected_rgb.ravel(), err_msg="PL input pixels")

    output_path = run / "native-reference-output.f32"
    tensor = np.fromfile(output_path, "<f4").reshape(MODEL.dense)
    require(output_path.stat().st_size == tensor.nbytes, "Raw output length is not an exact FP32 tensor")
    model_boxes = np.asarray(frame["model_boxes"], np.float32).reshape(-1, 6)
    source_boxes = np.asarray(frame["source_boxes"], np.float32).reshape(-1, 6)
    threshold = summary.get("score_threshold", 0.25)
    np.testing.assert_allclose(model_boxes, reference_boxes(tensor, threshold), rtol=0, atol=1e-6,
                               err_msg="Raw one-to-one boxes")
    np.testing.assert_array_equal(source_boxes, restore_boxes(model_boxes, width, height, left, top, gain),
                                  err_msg="Restored source boxes")
    require(summary["adopted_reference_boxes"] == len(source_boxes), "Sample box count differs from the run summary")

    composed_path = run / "native-reference-composed.rgb565"
    scanned_path = run / "native-reference-scan-buffer.rgb565"
    expected_display = compose_rgb565(rgb, source_boxes)
    for path in (composed_path, scanned_path):
        require(path.stat().st_size == expected_display.nbytes, "Incomplete display buffer: " + path.name)
        np.testing.assert_array_equal(np.fromfile(path, "<u2"), expected_display.ravel(), err_msg=path.name)
    return dict(epoch=frame["epoch"], frame=frame["frame"], boxes=len(source_boxes),
                adopted_address=frame["adopted_address"], adopted_tag=frame["adopted_hardware_tag"],
                sensor_to_adoption_ns=frame["adopted_ns"] - frame["captured_ns"],
                model_rgb_bytes=expected_rgb.nbytes, tensor_shape=MODEL.dense, tensor_bytes=tensor.nbytes,
                display_rgb565_bytes=expected_display.nbytes,
                sha256={path.name: sha256(path) for path in (source_path, rgb_path, output_path, composed_path,
                                                           scanned_path)}), expected_display


def analyze(run):
    def read(name):
        return json.loads((run / name).read_text())

    manifest, summary = read("contract.json"), read("native-live-summary.json")
    completion, outcome = read("native-complete.json"), read("outcome.json")
    require(manifest["schema"] == 4 and not summary["interrupted"], "Expected a completed rectangular video run")
    camera = read("preflight.json")["video_camera"]
    require(camera["usb_id"] == manifest["camera_usb_id"] and camera["device"] == summary["video_device"] and
            camera["bus_info"] == summary["video_bus_info"], "Captured camera differs from the preflight identity")
    count = validate_live_completion(completion, outcome["boot_id"], manifest, outcome["native_exit"], summary)
    require(count == outcome["completed"] and outcome["device_owners"] == [] and
            outcome["native_shutdown_complete"] is True and outcome["allocation_count_after"] == 0,
            "Run shutdown differs from native completion")
    sample, image = audit_sample(run, summary)
    stages = {}
    for name in STAGES:
        values = summary["metrics"]["stages"][name]
        windows = values["windows"]
        stages[name] = dict(new_frames=values["new_frames"], mean_fps=values["mean_fps"],
                            per_second_min=min(windows), per_second_max=max(windows),
                            windows_below_30=sum(value < 30 for value in windows))
    return dict(boot_id=outcome["boot_id"], BOOT_sha256=manifest["boot_sha256"],
                camera=camera, score_threshold=summary.get("score_threshold", 0.25),
                seconds=manifest["duration_seconds"], stages=stages, sample=sample,
                complete_chain_30fps_10min_passed=summary["complete_chain_30fps_10min_passed"],
                display_endpoint="adopted RGB565 DDR buffer", physical_screen_observed=False), image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, display = analyze(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    red, green, blue = display >> 11, (display >> 5) & 63, display & 31
    bgr = np.stack(((blue << 3) | (blue >> 2), (green << 2) | (green >> 4), (red << 3) | (red >> 2)), axis=-1)
    require(cv2.imwrite(str(args.output / "adopted-buffer.png"), bgr.astype(np.uint8)), "Cannot write RGB565 preview")
    save_json(args.output / "audit.json", result)
    print(json.dumps(result, indent=2))
