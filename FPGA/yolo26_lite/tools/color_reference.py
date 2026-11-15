"""生成像素解包／Q12 色彩转换向量，并用独立浮点矩阵和 OpenCV 校验。"""
from pathlib import Path
import numpy as np
import cv2
from common import save_json


COEFFICIENTS = ((4769, 6537, -1605, -3330, 8263),
                (4096, 5743, -1410, -2925, 7258),
                (4769, 7343, -873, -2183, 8652),
                (4096, 6450, -767, -1917, 7601))


def convert_q12(packed, matrix):
    pairs = packed.reshape(-1, 4).astype(np.int64)
    y = pairs[:, (0, 2)].reshape(-1)
    u = np.repeat(pairs[:, 1] - 128, 2)
    v = np.repeat(pairs[:, 3] - 128, 2)
    if matrix % 2 == 0:
        y = np.maximum(y - 16, 0)
    cy, cr, cgu, cgv, cb = COEFFICIENTS[matrix]
    values = np.stack((y * cy + v * cr, y * cy + u * cgu + v * cgv, y * cy + u * cb), axis=1)
    return np.clip((values + 2048) >> 12, 0, 255).astype(np.uint8)


def floating_reference(packed, matrix):
    pairs = packed.reshape(-1, 4).astype(np.float64)
    y = pairs[:, (0, 2)].reshape(-1)
    u = np.repeat(pairs[:, 1] - 128, 2)
    v = np.repeat(pairs[:, 3] - 128, 2)
    kr, kb = (0.299, 0.114) if matrix < 2 else (0.2126, 0.0722)
    kg = 1 - kr - kb
    if matrix % 2 == 0:
        y = np.maximum(y - 16, 0) * 255 / 219
        u *= 255 / 224
        v *= 255 / 224
    rgb = np.stack((y + 2 * (1 - kr) * v,
                    y - 2 * kb * (1 - kb) / kg * u - 2 * kr * (1 - kr) / kg * v,
                    y + 2 * (1 - kb) * u), axis=1)
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


def generate(directory):
    directory = Path(directory)
    rng = np.random.RandomState(0x260026)
    rows = []
    report = []
    for matrix in range(4):
        for width, height in ((2, 1), (6, 3), (130, 7), (640, 4)):
            packed = rng.randint(0, 256, width * height * 2, dtype=np.uint8)
            packed[:4] = [16, 128, 235, 128]
            expected = convert_q12(packed, matrix)
            floating = floating_reference(packed, matrix)
            error = int(np.max(np.abs(expected.astype(int) - floating.astype(int))))
            if error > 1:
                raise AssertionError("Q12 differs from colorimetry by more than one level")
            opencv_error = None
            if matrix == 0:
                opencv = cv2.cvtColor(packed.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2).reshape(-1, 3)
                opencv_error = int(np.max(np.abs(expected.astype(int) - opencv.astype(int))))
                if opencv_error > 1:
                    raise AssertionError("Q12 differs from OpenCV YUY2 by more than one level")
            rows.append((width, height, 0, matrix, packed, expected))
            report.append(dict(width=width, height=height, format="YUYV", matrix=matrix,
                               floating_max_abs=error, opencv_max_abs=opencv_error))
    for width, height in ((1, 1), (3, 2), (127, 5), (640, 4)):
        packed = rng.randint(0, 256, width * height * 3, dtype=np.uint8)
        rows.append((width, height, 1, 0, packed, packed.reshape(-1, 3)))
        report.append(dict(width=width, height=height, format="RGB", byte_exact=True))
    with (directory / "cases.txt").open("w", encoding="ascii") as cases:
        cases.write(str(len(rows)) + "\n")
        for index, (width, height, rgb, matrix, packed, expected) in enumerate(rows):
            cases.write("{} {} {} {} {}\n".format(index, width, height, rgb, matrix))
            (directory / ("input_{}.mem".format(index))).write_text(
                "\n".join("{:02x}".format(int(value)) for value in packed) + "\n", encoding="ascii")
            (directory / ("rgb_{}.mem".format(index))).write_text(
                "\n".join("{:02x}{:02x}{:02x}".format(*row) for row in expected) + "\n", encoding="ascii")
    save_json(directory / "color-reference.json", dict(seed="260026", cases=report))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    generate(arguments.output)
