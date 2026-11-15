"""生成 DDR → RGB → DDR 流水线向量，复用独立验证的数值参考。"""
from pathlib import Path
import cv2
import numpy as np
from color_reference import convert_q12
from resize_reference import resize_fixed
from common import save_json


def generate(directory, target=(640, 640)):
    directory = Path(directory)
    target_width, target_height = target
    rng = np.random.RandomState(0x26dd0001)
    report = []
    shapes = ((37, 29, True, 0), (640, 480, False, 0), (1280, 720, True, 0))
    with (directory / "cases.txt").open("w", encoding="ascii") as listing:
        listing.write(str(len(shapes)) + "\n")
        for index, (width, height, rgb, matrix) in enumerate(shapes):
            channels = 3 if rgb else 2
            source = rng.randint(0, 256, (height, width, channels), dtype=np.uint8)
            pixels = source if rgb else convert_q12(source, matrix).reshape(height, width, 3)
            gain = min(target_width / width, target_height / height)
            rw, rh = round(width * gain), round(height * gain)
            left, top = (target_width - rw) // 2, (target_height - rh) // 2
            resized, xm, ym = resize_fixed(pixels, rw, rh)
            independent = cv2.resize(pixels, (rw, rh), interpolation=cv2.INTER_LINEAR)
            error = int(np.max(np.abs(resized.astype(int) - independent.astype(int))))
            if error > 1:
                raise AssertionError("Integrated resize reference differs from OpenCV")
            expected = np.full((target_height, target_width, 3), 114, np.uint8)
            expected[top:top + rh, left:left + rw] = resized
            listing.write("{} {} {} {} {} {} {} {} {}\n".format(index, width, height, int(rgb),
                                                                  matrix, rw, rh, left, top))
            for name, array in (("source", source), ("expected", expected)):
                (directory / ("{}_{}.mem".format(name, index))).write_text(
                    "\n".join("{:02x}".format(int(value)) for value in array.reshape(-1)) + "\n", encoding="ascii")
            for name, mapping in (("x", xm), ("y", ym)):
                (directory / ("{}_{}.mem".format(name, index))).write_text(
                    "\n".join("{:06x}".format(int(lower) | (int(fraction) << 12)) for lower, fraction in mapping) + "\n",
                    encoding="ascii")
            report.append(dict(source=[width, height], format="RGB" if rgb else "YUYV", matrix=matrix,
                               resized=[rw, rh], padding=[left, top], opencv_max_abs=error))
    save_json(directory / "preprocess-reference.json", dict(target=list(target), seed="26dd0001", cases=report))
