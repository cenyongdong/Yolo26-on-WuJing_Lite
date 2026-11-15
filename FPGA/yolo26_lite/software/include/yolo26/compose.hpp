#pragma once
#include "yolo26/frames.hpp"
#include "yolo26/preprocess.hpp"
#include <array>
#include <vector>

namespace yolo26 {

// 构造时核对推理身份并还原坐标；之后原图与框不可分别替换。
// 原图引用持续占有采集槽位，直到推理、合成等全部消费者归还。
class DetectionBundle {
public:
    DetectionBundle(FrameRef frame, FrameKey inference_key, const Letterbox& model_geometry,
                    ColorMatrix matrix, std::vector<Detection> model_boxes, std::uint64_t completed_ns);
    const FrameRef& frame() const { return frame_; }
    const std::vector<Detection>& boxes() const { return boxes_; }
    ColorMatrix matrix() const { return matrix_; }
    std::uint64_t completed_ns() const { return completed_ns_; }

private:
    FrameRef frame_;
    std::vector<Detection> boxes_;
    ColorMatrix matrix_;
    std::uint64_t completed_ns_;
};

// 普通 Host 内存合成；单一显示消费者拥有此对象。固定输出 1280x720 RGB565 小端。
// 保持宽高比补黑边，最近邻只用于显示；模型的双线性预处理不受影响。
// 池满返回空，不等待扫描、不覆盖已发布帧；硬件显示缓冲另由 SDK 适配层持有。
class Compositor720p {
public:
    static constexpr std::size_t kBytes = 1280 * 720 * 2;
    FrameRef compose(const DetectionBundle& detection, FramePool& display_pool);

private:
    void configure(const FrameInfo& source);
    std::uint32_t source_width_ = 0, source_height_ = 0;
    Letterbox geometry_{};
    std::array<std::uint16_t, 1280> source_row_{};
    std::array<std::uint32_t, 1280> horizontal_{};
    std::array<std::uint32_t, 720> vertical_{};
};

} // namespace yolo26
