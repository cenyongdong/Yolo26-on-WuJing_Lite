"""建立固定且按内容隔离的校准／留出集；复用 PL 的 Q12 缩放参考，不访问设备。"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import save_json, sha256
from model_geometry import ModelGeometry, SQUARE
from resize_reference import resize_fixed


def partition(records, calibration_count, seed):
    """同内容图片不能跨集合；排序只依赖内容和种子，不受目录枚举顺序影响。"""
    if not 0 < calibration_count < len(records):
        raise ValueError("Both calibration and held-out sets must be nonempty")
    if len({row["sha256"] for row in records}) != len(records):
        raise ValueError("Duplicate image content would leak across the split")
    ordered = sorted(records, key=lambda row: hashlib.sha256(
        (seed + ":" + row["sha256"]).encode("utf-8")).hexdigest())
    return ordered[:calibration_count], ordered[calibration_count:]


def prepare_pixels(bgr, target=640):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected a decoded three-channel UInt8 image")
    height, width = bgr.shape[:2]
    target_width, target_height = (target, target) if isinstance(target, int) else target
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("Invalid image geometry")
    gain = min(target_width / width, target_height / height)
    resized_width, resized_height = max(1, round(width * gain)), max(1, round(height * gain))
    left, top = (target_width - resized_width) // 2, (target_height - resized_height) // 2
    resized, _, _ = resize_fixed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resized_width, resized_height)
    rgb = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    rgb[top:top + resized_height, left:left + resized_width] = resized
    normalized = rgb.astype(np.float32) / np.float32(255)
    return rgb, normalized, dict(original=[width, height], resized=[resized_width, resized_height],
                                 padding=[left, top], scale_xy=[resized_width / width, resized_height / height])


def read_labels(path):
    # COCO128 允许没有标注文件的背景图；缺失事实单独记入清单，不凭空生成目标。
    if not path.exists():
        return [], False
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if (len(values) != 5 or not np.isfinite(values).all() or values[0] != int(values[0]) or
                not 0 <= values[0] < 80 or not all(0 <= value <= 1 for value in values[1:]) or
                min(values[3:]) <= 0):
            raise ValueError("Invalid normalized YOLO label in " + str(path))
        rows.append(values)
    return rows, True


def prepare(images, labels, output, calibration_count=96, seed="yolo26-int8-r1", geometry=SQUARE):
    paths = sorted(path for path in images.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    records = [dict(image=path.name, sha256=sha256(path)) for path in paths]
    calibration, validation = partition(records, calibration_count, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration").mkdir()
    (output / "validation").mkdir()
    for group, rows in (("calibration", calibration), ("validation", validation)):
        for index, row in enumerate(rows):
            rgb, normalized, transform = prepare_pixels(cv2.imread(str(images / row["image"])),
                                                         (geometry.width, geometry.height))
            stem = "case-{}".format(index)
            directory = output / group
            if group == "calibration":
                name = stem + ".ftmp"
                normalized.astype("<f4").tofile(str(directory / name))
            else:
                name = stem + "-input.npy"
                np.save(str(directory / name), normalized.transpose(2, 0, 1)[None])
                rgb.tofile(str(directory / (stem + ".rgb")))
            label_path = labels / (Path(row["image"]).stem + ".txt")
            boxes, present = read_labels(label_path)
            row.update(case=index, input=name, input_sha256=sha256(directory / name),
                       transform=transform, labels_yolo=boxes, label_file_present=present,
                       label_sha256=sha256(label_path) if present else None)
    lists = {}
    for count in sorted({min(4, len(calibration)), min(32, len(calibration)), len(calibration)}):
        name = "calibration-{}.txt".format(count)
        (output / name).write_text("\n".join(row["input"] for row in calibration[:count]) + "\n", encoding="ascii")
        lists[str(count)] = dict(path=name, sha256=sha256(output / name))
    manifest = dict(schema=1, source_images=str(images.resolve()), source_labels=str(labels.resolve()),
                    seed=seed, target=[geometry.width, geometry.height], input_layout="NHWC",
                    input_dtype="FP32", rgb_order=True,
                    normalization="FP32 RGB / 255 exactly once", resize="half_pixel_Q12_reference",
                    padding_value=114, calibration=calibration, validation=validation, lists=lists,
                    content_disjoint=True, task_scene_representative=False,
                    labelled_proxy_only=True, official_coco_validation=False, board_access=False)
    save_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=96)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    manifest = prepare(args.images, args.labels, args.output, args.calibration_count,
                       geometry=ModelGeometry(*args.size))
    print("Prepared", len(manifest["calibration"]), "calibration and", len(manifest["validation"]),
          "content-disjoint held-out images; not official COCO validation")


if __name__ == "__main__":
    main()
