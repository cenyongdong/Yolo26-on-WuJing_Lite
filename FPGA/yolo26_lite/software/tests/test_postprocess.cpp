#include "yolo26/postprocess.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    using namespace yolo26;
    std::vector<float> raw(kRawElements, -100.0F);
    std::fill(raw.begin(), raw.begin() + 4 * kAnchors, 1.0F);
    // 边界锚点分别落在三个尺度，且同一锚点允许保留多个类别。
    raw[(4 + 3) * kAnchors] = 5.0F;
    raw[(4 + 4) * kAnchors] = 4.0F;
    raw[(4 + 9) * kAnchors + 6400] = 3.0F;
    raw[(4 + 79) * kAnchors + 8000] = 2.0F;
    auto boxes = decode_one_to_one(raw.data(), raw.size());
    require(boxes.size() == 4, "threshold selection");
    require(boxes[0].class_id == 3 && boxes[1].class_id == 4, "same anchor multi-class semantics");
    require(boxes[0].x1 == -4 && boxes[0].x2 == 12, "P3 anchor decoding");
    require(boxes[2].x1 == -8 && boxes[2].x2 == 24, "P4 anchor decoding");
    require(boxes[3].x1 == -16 && boxes[3].x2 == 48, "P5 anchor decoding");
    require(decode_one_to_one(raw.data(), raw.size(), 1.0F).empty(), "upper threshold");
    require(decode_one_to_one(raw.data(), raw.size(), 0.0F, 1).size() == 1, "bounded max_det");
    auto tied = raw;
    std::fill(tied.begin() + 4 * kAnchors, tied.end(), 0.0F);
    const auto ties = decode_one_to_one(tied.data(), tied.size());
    require(ties.size() == 300 && ties[0].class_id == 0 && ties[80].class_id == 0 &&
            ties[80].x1 == 4.0F && ties[299].class_id == 59, "stable equal-score selection");
    // 覆盖向量的四条 lane、坐标/分类边界及最后一个未入选值；低分垃圾仍必须隔离。
    const std::size_t invalid_positions[] = {0, 1, 2, 3, 15, 16, 31, 32, 4 * kAnchors - 1,
                                             4 * kAnchors, 17 * kAnchors + 7, 17 * kAnchors + 15,
                                             17 * kAnchors + 16, kRawElements - 1};
    for (const auto position : invalid_positions) {
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()}) {
            const auto original = tied[position];
            tied[position] = invalid;
            bool rejected = false;
            try {
                decode_one_to_one(tied.data(), tied.size());
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            tied[position] = original;
            require(rejected, "non-finite tensor lane escaped quarantine");
        }
    }
    for (int scenario = 0; scenario < 6; ++scenario) {
        bool rejected = false;
        try {
            if (scenario == 0) decode_one_to_one(nullptr, raw.size());
            if (scenario == 1) decode_one_to_one(raw.data(), raw.size() - 1);
            if (scenario == 2) decode_one_to_one(raw.data(), raw.size(), -0.1F);
            if (scenario == 3) decode_one_to_one(raw.data(), raw.size(), 0.25F, 301);
            if (scenario == 4) {
                raw.back() = std::numeric_limits<float>::quiet_NaN();
                decode_one_to_one(raw.data(), raw.size());
            }
            if (scenario == 5) {
                raw.back() = -100.0F;
                raw[0] = std::numeric_limits<float>::max();
                decode_one_to_one(raw.data(), raw.size());
            }
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid input must be quarantined");
    }
    auto transform = make_letterbox(640, 480);
    require(transform.resized_width == 640 && transform.top == 80 && transform.gain == 1.0F,
            "VGA letterbox");
    transform = make_letterbox(1280, 481);
    require(transform.resized_height == 240 && transform.top == 200, "ties-to-even resize");
    Detection box{160, 200, 480, 440, 0.9F, 0};
    require(restore_box(box, transform), "valid restored box");
    require(box.x1 == 320 && box.y1 == 0 && box.x2 == 960 && box.y2 == 480, "coordinate restore");
    box.x2 = box.x1;
    require(!restore_box(box, make_letterbox(640, 640)), "degenerate box must be hidden");
    // 512×384 下 P3 每行 64 个位置。验证换行位置与两个尺度交界处的独立坐标。
    const auto rectangular = kRectangularModel;
    require(rectangular.anchors() == 4032, "rectangular anchor count");
    std::vector<float> rectangle_raw(rectangular.raw_elements(), -100.0F);
    std::fill_n(rectangle_raw.begin(), 4 * 4032, 1.0F);
    rectangle_raw[(4 + 3) * 4032 + 64] = 5.0F;
    rectangle_raw[(4 + 4) * 4032 + 3072] = 4.0F;
    rectangle_raw[(4 + 5) * 4032 + 4031] = 3.0F;
    const auto rectangle_boxes = decode_one_to_one(rectangle_raw.data(), rectangle_raw.size(), 0.25F, 300, rectangular);
    require(rectangle_boxes.size() == 3 && rectangle_boxes[0].x1 == -4 && rectangle_boxes[0].y1 == 4 &&
            rectangle_boxes[1].x1 == -8 && rectangle_boxes[1].y1 == -8 &&
            rectangle_boxes[2].x1 == 464 && rectangle_boxes[2].y1 == 336, "rectangular per-scale coordinates");
    // 阈值为零时不预先裁剪。以该路径选出的完整结果检查阈值过滤、并列和极小分值。
    rectangle_raw[4 * 4032] = -103.0F;
    rectangle_raw[5 * 4032] = std::nextafter(-1.0986123F, 0.0F);
    rectangle_raw[6 * 4032] = -1.0986123F;
    rectangle_raw[7 * 4032] = std::nextafter(-1.0986123F, -100.0F);
    rectangle_raw[8 * 4032] = 128.0F;
    const auto unpruned = decode_one_to_one(rectangle_raw.data(), rectangle_raw.size(), 0.0F, 300, rectangular);
    for (const float threshold : {std::numeric_limits<float>::denorm_min(), 0.25F, 0.5F, 0.999F, 1.0F}) {
        std::vector<Detection> expected;
        std::copy_if(unpruned.begin(), unpruned.end(), std::back_inserter(expected),
                     [threshold](const Detection& entry) { return entry.score >= threshold; });
        const auto selected = decode_one_to_one(rectangle_raw.data(), rectangle_raw.size(), threshold, 300, rectangular);
        require(selected.size() == expected.size() &&
                std::equal(selected.begin(), selected.end(), expected.begin(), [](const Detection& a, const Detection& b) {
                    return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2 &&
                           a.score == b.score && a.class_id == b.class_id;
                }), "early pruning changed the selected detections");
    }
    std::cout << "PASS postprocess: scales, two-stage selection, invalid input, geometry\n";
}
