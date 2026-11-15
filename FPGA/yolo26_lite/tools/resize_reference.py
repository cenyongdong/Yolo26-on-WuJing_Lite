"""生成 half-pixel Q12 映射与独立 OpenCV 校验向量。"""
from pathlib import Path
import cv2
import numpy as np
from common import save_json


def axis_map(source_size, resized_size):
    entries = []
    denominator = 2 * resized_size
    for output_index in range(resized_size):
        numerator = (2 * output_index + 1) * source_size - resized_size
        if numerator <= 0:
            lower, fraction = 0, 0
        elif numerator >= (source_size - 1) * denominator:
            lower, fraction = source_size - 1, 0
        else:
            lower, remainder = divmod(numerator, denominator)
            fraction = (remainder * 4096 + denominator // 2) // denominator
            if fraction == 4096:
                lower += 1
                fraction = 0
        entries.append((lower, fraction))
    return np.asarray(entries, dtype=np.int64)


def resize_fixed(image, width, height):
    xm = axis_map(image.shape[1], width)
    ym = axis_map(image.shape[0], height)
    x0, fx = xm[:, 0], xm[:, 1]
    y0, fy = ym[:, 0], ym[:, 1]
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    pixels = image.astype(np.int64)
    upper = pixels[y0[:, None], x0[None, :]] * (4096 - fx)[None, :, None]
    upper += pixels[y0[:, None], x1[None, :]] * fx[None, :, None]
    lower = pixels[y1[:, None], x0[None, :]] * (4096 - fx)[None, :, None]
    lower += pixels[y1[:, None], x1[None, :]] * fx[None, :, None]
    result = (upper * (4096 - fy)[:, None, None] + lower * fy[:, None, None] + (1 << 23)) >> 24
    return result.astype(np.uint8), xm, ym


def generate(directory):
    directory = Path(directory)
    target_width = target_height = 16
    shapes = ((4, 4), (32, 24), (17, 13), (1, 5), (47, 31), (1280, 80),
              (80, 720), (1, 1), (16, 16), (16, 9))
    rng = np.random.RandomState(0x261600)
    report = []
    with (directory / "cases.txt").open("w", encoding="ascii") as cases:
        cases.write(str(len(shapes)) + "\n")
        for index, (width, height) in enumerate(shapes):
            image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
            gain = min(target_width / width, target_height / height)
            rw, rh = round(width * gain), round(height * gain)
            left, top = (target_width - rw) // 2, (target_height - rh) // 2
            resized, xm, ym = resize_fixed(image, rw, rh)
            independent = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
            error = int(np.max(np.abs(resized.astype(int) - independent.astype(int))))
            if error > 1:
                raise AssertionError("Fixed resize differs from OpenCV by more than one level")
            padded = np.full((target_height, target_width, 3), 114, np.uint8)
            padded[top:top + rh, left:left + rw] = resized
            cases.write("{} {} {} {} {} {} {}\n".format(index, width, height, rw, rh, left, top))
            for name, pixels in (("source", image), ("expected", padded)):
                (directory / ("{}_{}.mem".format(name, index))).write_text(
                    "\n".join("{:02x}{:02x}{:02x}".format(*pixel) for pixel in pixels.reshape(-1, 3)) + "\n",
                    encoding="ascii")
            for name, mapping in (("x", xm), ("y", ym)):
                (directory / ("{}_{}.mem".format(name, index))).write_text(
                    "\n".join("{:06x}".format(int(lower) | (int(fraction) << 12)) for lower, fraction in mapping) + "\n",
                    encoding="ascii")
            report.append(dict(source=[width, height], resized=[rw, rh], padding=[left, top], opencv_max_abs=error))
    save_json(directory / "resize-reference.json", dict(target=[16, 16], seed="261600", cases=report))
