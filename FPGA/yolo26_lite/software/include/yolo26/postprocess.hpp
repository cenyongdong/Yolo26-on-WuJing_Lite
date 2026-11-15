#pragma once
#include "yolo26/model_geometry.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolo26 {

constexpr std::size_t kAnchors = kSquareModel.anchors();
constexpr std::size_t kRawElements = kSquareModel.raw_elements();

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    std::uint32_t class_id;
};

// 输入为 one-to-one [1,84,anchors] FP32，前 4 通道 LTRB，其后为 80 类 logits。
// 不执行 IoU 抑制。不借用 tensor 生命周期；非法长度／非有限值抛出 invalid_argument。
// 分值并列时按候选位置稳定选取；允许与 PyTorch 的非稳定并列索引不同。
std::vector<Detection> decode_one_to_one(const float* tensor, std::size_t elements,
                                         float threshold = 0.25F, std::size_t max_det = 300,
                                         ModelGeometry geometry = kSquareModel);

// 仅报告张量扫描的实现路径，便于保存性能证据；不改变后处理数学或检测接口。
const char* postprocess_scan_backend() noexcept;

struct Letterbox {
    std::uint32_t source_width;
    std::uint32_t source_height;
    std::uint32_t target_width;
    std::uint32_t target_height;
    std::uint32_t resized_width;
    std::uint32_t resized_height;
    std::uint32_t left;
    std::uint32_t top;
    float gain;
};

// 使用 ties-to-even 整数舍入，与固定尺寸 Ultralytics Letterbox 一致。
Letterbox make_letterbox(std::uint32_t source_width, std::uint32_t source_height,
                         std::uint32_t target_width = 640, std::uint32_t target_height = 640);

// 将模型坐标还原到原画面并裁剪。返回 false 的退化／非有限框不能发布或绘制。
bool restore_box(Detection& detection, const Letterbox& transform);

} // namespace yolo26
