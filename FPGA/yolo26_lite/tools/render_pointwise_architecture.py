"""从已核验的位置映射绘制可审阅的算法架构图；时间只统计组件内的 1×1。"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from analyze_npu import read_json, require


def render(record, font_path, output):
    groups = {group["module"]: group for group in record["modules"]}
    image = Image.new("RGB", (2100, 1500), "#f7f9fc")
    drawing = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (22, 27, 31, 45)}
    boxes = {}

    def box(key, x, y, width, height, lines, hot=False, host=False):
        fill, stroke = ("#fff0df", "#b45c12") if hot else ("#e7f2ff", "#648db8")
        if host:
            fill, stroke = "#eef0f4", "#858e9b"
        drawing.rounded_rectangle((x, y, x + width, y + height), radius=18, fill=fill, outline=stroke, width=3)
        total_height = 34 * len(lines)
        for index, text in enumerate(lines):
            font = fonts[27]
            bounds = drawing.textbbox((0, 0), text, font=font)
            require(bounds[2] - bounds[0] <= width - 18, "Diagram label exceeds its box")
            drawing.text((x + (width - bounds[2]) / 2, y + (height - total_height) / 2 + index * 34),
                         text, font=font, fill="#163653")
        boxes[key] = (x, y, width, height)

    def module(key, x, y, identity, title, detail=None):
        group = groups[identity]
        lines = [title, "1×1：%d 层，合计 %.3f ms" % (group["count"], group["mean_ms"])]
        if detail:
            lines.append(detail)
        box(key, x, y, 550 if x == 770 else 510, 132, lines, hot=bool(detail))

    def arrow(points):
        drawing.line(points, fill="#73879c", width=3, joint="curve")
        x, y = points[-1]
        previous_x, previous_y = points[-2]
        angle = math.atan2(y - previous_y, x - previous_x)
        drawing.polygon([(x, y), (x - 16 * math.cos(angle - .38), y - 16 * math.sin(angle - .38)),
                         (x - 16 * math.cos(angle + .38), y - 16 * math.sin(angle + .38))], fill="#73879c")

    drawing.text((70, 28), "YOLO26l：1×1 卷积位置与慢层", font=fonts[45], fill="#102d4a")
    drawing.text((70, 94), "原图 FP16 / 640×640 / batch=1　　86 个 1×1，合计 135.022 ms", font=fonts[31], fill="#375873")
    drawing.text((70, 140), "橙色框：前三个慢层所在组件。框中时间只含 1×1，不代表整个组件耗时。", font=fonts[22], fill="#5c6b7b")
    module("b2", 70, 190, 2, "Backbone · model.2 · 160×160", "cv2 / #3223：15.192 ms")
    module("b4", 70, 370, 4, "Backbone · model.4 · 80×80", "cv2 / #3240：18.794 ms")
    module("b6", 70, 550, 6, "Backbone · model.6 · 40×40")
    module("b8", 70, 730, 8, "Backbone · model.8 · 20×20")
    module("spp", 70, 910, 9, "SPPF · model.9")
    module("psa", 70, 1090, 10, "C2PSA · model.10 · 两组注意力")
    module("n13", 770, 200, 13, "Neck · model.13 · 40×40")
    module("n16", 770, 420, 16, "Neck P3 · model.16 · 80×80", "cv1 / #3296：19.040 ms")
    module("n19", 770, 640, 19, "Neck P4 · model.19 · 40×40")
    module("n22", 770, 860, 22, "Neck P5 · model.22 · 20×20", None)
    drawing.text((780, 1004), "model.22 内含第三组注意力", font=fonts[22], fill="#5c6b7b")
    box("head", 1480, 420, 550, 240, ["one-to-one 检测头 · model.23", "1×1：12 层，合计 11.305 ms",
        "通道投影＋最终输出投影", "每位置：4 个 LTRB＋80 类 logits"])
    box("raw", 1480, 750, 550, 110, ["原始输出 1×84×8400", "SDK 读回及布局恢复"])
    box("post", 1480, 960, 550, 174, ["PS：图外尾处理（非 NPU）", "两阶段 TopK / Gather", "sigmoid / 解码 / 阈值", "无 NMS"], host=True)
    box("result", 1480, 1230, 550, 100, ["最多 300 个框", "恢复坐标并与同帧画面合成"], host=True)
    for first, second in zip(("b2", "b4", "b6", "b8", "spp"), ("b4", "b6", "b8", "spp", "psa")):
        x, y, w, h = boxes[first]
        arrow([(x + w / 2, y + h), (x + w / 2, boxes[second][1])])
    arrow([(580, 616), (670, 616), (670, 266), (770, 266)])
    arrow([(580, 436), (715, 436), (715, 486), (770, 486)])
    arrow([(580, 1156), (630, 1156), (630, 202), (770, 202)])
    arrow([(580, 1190), (710, 1190), (710, 926), (770, 926)])
    for first, second in (("n13", "n16"), ("n16", "n19"), ("n19", "n22")):
        x, y, w, h = boxes[first]
        arrow([(x + w / 2, y + h), (x + w / 2, boxes[second][1])])
    arrow([(1320, 266), (1360, 266), (1360, 706), (1320, 706)])
    arrow([(1320, 486), (1480, 486)])
    arrow([(1320, 738), (1420, 738), (1420, 546), (1480, 546)])
    arrow([(1320, 926), (1450, 926), (1450, 606), (1480, 606)])
    for first, second in (("head", "raw"), ("raw", "post"), ("post", "result")):
        x, y, w, h = boxes[first]
        arrow([(x + w / 2, y + h), (x + w / 2, boxes[second][1])])
    drawing.text((70, 1320), "省略部分 3×3 下采样细节；Neck 连线包含上／下采样与 Concat。", font=fonts[27], fill="#526a80")
    drawing.text((70, 1372), "计时：原生完成层（含计算及内部搬运）；三个重点层合计约 53.027 ms。", font=fonts[27], fill="#526a80")
    drawing.text((70, 1422), "位置依据：编译器输出名映射；数据：evidence/pointwise-location-r1.json", font=fonts[22], fill="#526a80")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(read_json(args.mapping), args.font, args.output)


if __name__ == "__main__":
    main()
