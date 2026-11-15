#include "yolo26/postprocess.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
#include <arm_neon.h>
#endif

namespace yolo26 {
namespace {

std::array<float, kAnchors> checked_class_maxima(const float* tensor, std::size_t anchor_count) {
    std::array<float, kAnchors> maxima;
    std::fill_n(maxima.begin(), anchor_count, -std::numeric_limits<float>::infinity());
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
    // 宽高均为 32 的倍数，P5 位置数仍可能不整除四，尾项须检查。
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "FP32 tensor required");
    // 连续四个锚点一组；检查 IEEE754 指数是否全 1，同时覆盖坐标和未入选类别。
    // 先累积错误掩码，整次扫描后统一隔离。NaN/Inf 永远不会进入 TopK 或结果发布。
    const auto exponent = vdupq_n_u32(0x7f800000U);
    auto invalid = vdupq_n_u32(0);
    for (std::size_t channel = 0; channel < 4 + kClasses; ++channel) {
        const float* values = tensor + channel * anchor_count;
        for (std::size_t anchor = 0; anchor + 4 <= anchor_count; anchor += 4) {
            const auto value = vld1q_f32(values + anchor);
            const auto bits = vreinterpretq_u32_f32(value);
            invalid = vorrq_u32(invalid, vceqq_u32(vandq_u32(bits, exponent), exponent));
            if (channel >= 4) {
                vst1q_f32(maxima.data() + anchor, vmaxq_f32(vld1q_f32(maxima.data() + anchor), value));
            }
        }
    }
    bool valid = vmaxvq_u32(invalid) == 0;
    for (std::size_t channel = 0; channel < kRawChannels; ++channel) {
        const float* values = tensor + channel * anchor_count;
        for (std::size_t anchor = anchor_count & ~std::size_t{3}; anchor < anchor_count; ++anchor) {
            valid = std::isfinite(values[anchor]) && valid;
            if (channel >= 4) {
                maxima[anchor] = std::max(maxima[anchor], values[anchor]);
            }
        }
    }
#else
    // 通用平台保留相同检查和访问顺序；只把异常出口移到循环外，允许编译器向量化。
    bool valid = true;
    for (std::size_t channel = 0; channel < 4 + kClasses; ++channel) {
        const float* values = tensor + channel * anchor_count;
        for (std::size_t anchor = 0; anchor < anchor_count; ++anchor) {
            valid = std::isfinite(values[anchor]) && valid;
            if (channel >= 4) {
                maxima[anchor] = std::max(maxima[anchor], values[anchor]);
            }
        }
    }
#endif
    if (!valid) {
        throw std::invalid_argument("non-finite NPU output quarantined");
    }
    return maxima;
}

struct Candidate {
    float logit;
    std::size_t index;
};

bool higher(const Candidate& left, const Candidate& right) {
    if (left.logit != right.logit) {
        return left.logit > right.logit;
    }
    return left.index < right.index;
}

void keep_top(std::vector<Candidate>& candidates, std::size_t count) {
    if (count < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(), higher);
        candidates.resize(count);
    }
    std::sort(candidates.begin(), candidates.end(), higher);
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

std::uint32_t rounded_ratio(std::uint64_t numerator, std::uint32_t denominator) {
    const auto quotient = numerator / denominator;
    const auto remainder_twice = (numerator % denominator) * 2;
    return static_cast<std::uint32_t>(quotient +
        (remainder_twice > denominator || (remainder_twice == denominator && (quotient & 1))));
}

} // namespace

const char* postprocess_scan_backend() noexcept {
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
    return "aarch64_neon";
#else
    return "portable_scalar";
#endif
}

std::vector<Detection> decode_one_to_one(const float* tensor, std::size_t elements,
                                         float threshold, std::size_t max_det, ModelGeometry geometry) {
    if (!geometry.valid() || !tensor || elements != geometry.raw_elements() || !std::isfinite(threshold) ||
        threshold < 0.0F || threshold > 1.0F || max_det == 0 || max_det > 300 || max_det > geometry.anchors()) {
        throw std::invalid_argument("invalid one-to-one tensor or selection parameters");
    }
    const auto anchor_count = geometry.anchors();
    const auto maxima = checked_class_maxima(tensor, anchor_count);
    // 负 logit 的 sigmoid 不超过 exp(logit)。上界留出四倍余量，
    // 包括阈值为 FP32 次正规数时的舍入；临界候选仍按原 sigmoid 判定。
    const double cutoff = threshold == 0.0F ? -std::numeric_limits<double>::infinity()
                                             : std::log(static_cast<double>(threshold) * 0.25);
    std::vector<Candidate> anchors;
    anchors.reserve(anchor_count);
    for (std::size_t anchor = 0; anchor < anchor_count; ++anchor) {
        if (maxima[anchor] >= cutoff) {
            anchors.push_back({maxima[anchor], anchor});
        }
    }
    keep_top(anchors, max_det);

    std::vector<Candidate> classes;
    classes.reserve(anchors.size() * kClasses);
    for (std::size_t slot = 0; slot < anchors.size(); ++slot) {
        for (std::size_t class_id = 0; class_id < kClasses; ++class_id) {
            classes.push_back({tensor[(4 + class_id) * anchor_count + anchors[slot].index],
                               slot * kClasses + class_id});
        }
    }
    keep_top(classes, max_det);

    // Sigmoid 单调，因此两阶段 TopK 在 logit 域执行，仅为选出的最多 300 项计算 exp。
    // 浮点 sigmoid 饱和引入的并列属于允许的 TopK 并列，不改变分值阈值语义。
    std::vector<Detection> detections;
    detections.reserve(max_det);
    for (const auto& item : classes) {
        const float score = sigmoid(item.logit);
        if (score < threshold) {
            continue;
        }
        const auto anchor = anchors[item.index / kClasses].index;
        std::size_t local_index = anchor;
        auto stride = kDetectionStrides.front();
        auto grid_width = geometry.width / stride;
        for (const auto level_stride : kDetectionStrides) {
            stride = level_stride;
            grid_width = geometry.width / stride;
            const auto grid_size = std::size_t{grid_width} * (geometry.height / stride);
            if (local_index < grid_size) {
                break;
            }
            local_index -= grid_size;
        }
        const float center_x = static_cast<float>(local_index % grid_width) + 0.5F;
        const float center_y = static_cast<float>(local_index / grid_width) + 0.5F;
        const Detection detection{
            (center_x - tensor[anchor]) * stride,
            (center_y - tensor[anchor_count + anchor]) * stride,
            (center_x + tensor[2 * anchor_count + anchor]) * stride,
            (center_y + tensor[3 * anchor_count + anchor]) * stride,
            score, static_cast<std::uint32_t>(item.index % kClasses)};
        if (!std::isfinite(detection.x1) || !std::isfinite(detection.y1) ||
            !std::isfinite(detection.x2) || !std::isfinite(detection.y2)) {
            throw std::invalid_argument("decoded coordinates overflowed");
        }
        detections.push_back(detection);
    }
    return detections;
}

Letterbox make_letterbox(std::uint32_t source_width, std::uint32_t source_height,
                         std::uint32_t target_width, std::uint32_t target_height) {
    // 硬件支持的几何上限同时限制乘法和行缓存；不接受隐式裁切。
    if (!source_width || !source_height || source_width > 1280 || source_height > 720 ||
        !target_width || !target_height || target_width > 1280 || target_height > 720) {
        throw std::invalid_argument("image geometry outside supported limits");
    }
    const bool width_limited = std::uint64_t(target_width) * source_height <=
                               std::uint64_t(target_height) * source_width;
    const auto numerator = width_limited ? target_width : target_height;
    const auto denominator = width_limited ? source_width : source_height;
    const auto resized_width = rounded_ratio(std::uint64_t(source_width) * numerator, denominator);
    const auto resized_height = rounded_ratio(std::uint64_t(source_height) * numerator, denominator);
    if (!resized_width || !resized_height) {
        throw std::invalid_argument("resized image is empty");
    }
    return {source_width, source_height, target_width, target_height, resized_width, resized_height,
            (target_width - resized_width) / 2, (target_height - resized_height) / 2,
            static_cast<float>(numerator) / denominator};
}

bool restore_box(Detection& detection, const Letterbox& transform) {
    if (!std::isfinite(detection.x1) || !std::isfinite(detection.y1) ||
        !std::isfinite(detection.x2) || !std::isfinite(detection.y2) ||
        !std::isfinite(detection.score) || detection.score < 0.0F || detection.score > 1.0F ||
        detection.class_id >= kClasses || !transform.source_width || !transform.source_height ||
        !std::isfinite(transform.gain) || transform.gain <= 0.0F) {
        return false;
    }
    detection.x1 = std::clamp((detection.x1 - transform.left) / transform.gain, 0.0F,
                              static_cast<float>(transform.source_width));
    detection.x2 = std::clamp((detection.x2 - transform.left) / transform.gain, 0.0F,
                              static_cast<float>(transform.source_width));
    detection.y1 = std::clamp((detection.y1 - transform.top) / transform.gain, 0.0F,
                              static_cast<float>(transform.source_height));
    detection.y2 = std::clamp((detection.y2 - transform.top) / transform.gain, 0.0F,
                              static_cast<float>(transform.source_height));
    return detection.x1 < detection.x2 && detection.y1 < detection.y2;
}

} // namespace yolo26
