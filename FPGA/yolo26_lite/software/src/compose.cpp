#include "yolo26/compose.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
#include <arm_neon.h>
#endif

namespace yolo26 {
namespace {

constexpr std::size_t kDisplayStride = 1280 * 2;
constexpr std::uint16_t kPalette[] = {0x07e0, 0xf800, 0x001f, 0xffe0, 0x07ff, 0xf81f, 0xffff, 0xfd20};
// 与 PL 的 Q12 颜色矩阵相同；在 CPU 端只执行画面显示所需的颜色转换。
constexpr int kCoefficients[4][5] = {
    {4769, 6537, -1605, -3330, 8263}, {4096, 5743, -1410, -2925, 7258},
    {4769, 7343, -873, -2183, 8652}, {4096, 6450, -767, -1917, 7601}};

std::uint8_t clip_q12(int value) {
    // 先裁剪再右移，避免 C++17 对负有符号数移位的实现相关行为。
    return static_cast<std::uint8_t>(std::clamp(value + 2048, 0, 255 * 4096) / 4096);
}

std::uint16_t rgb565(std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    return static_cast<std::uint16_t>((red >> 3) << 11 | (green >> 2) << 5 | (blue >> 3));
}

#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
uint16x8_t rgb565(uint16x8_t red, uint16x8_t green, uint16x8_t blue) {
    return vorrq_u16(vorrq_u16(vshlq_n_u16(vshrq_n_u16(red, 3), 11),
                               vshlq_n_u16(vshrq_n_u16(green, 2), 5)), vshrq_n_u16(blue, 3));
}

struct ColorQ12 {
    int32x4_t red, green, blue;
};

ColorQ12 color_four(int16x4_t y, int16x4_t u, int16x4_t v, const int* coefficients) {
    const auto luma = vmull_n_s16(y, coefficients[0]);
    return {vmlal_n_s16(luma, v, coefficients[1]),
            vmlal_n_s16(vmlal_n_s16(luma, u, coefficients[2]), v, coefficients[3]),
            vmlal_n_s16(luma, u, coefficients[4])};
}

uint16x8_t clip_eight(int32x4_t lower, int32x4_t upper) {
    // 有符号 Q12 加半再右移、饱和到非负，与标量 clip_q12 的整数舍入逐位相同。
    // 第一层缩窄仅饱和到 16bit，所以还必须把大于 255 的值裁回 8bit 范围。
    return vminq_u16(vcombine_u16(vqrshrun_n_s32(lower, 12), vqrshrun_n_s32(upper, 12)), vdupq_n_u16(255));
}
#endif

void decode_row(const std::uint8_t* input, std::uint32_t width, PixelFormat format,
                ColorMatrix matrix, std::uint16_t* output) {
    std::uint32_t x = 0;
    if (format == PixelFormat::Rgb888) {
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
        for (; x + 8 <= width; x += 8) {
            const auto channels = vld3_u8(input + x * 3);
            vst1q_u16(output + x, rgb565(vmovl_u8(channels.val[0]), vmovl_u8(channels.val[1]),
                                        vmovl_u8(channels.val[2])));
        }
#endif
        for (; x < width; ++x) {
            output[x] = rgb565(input[x * 3], input[x * 3 + 1], input[x * 3 + 2]);
        }
        return;
    }
    const auto* coefficients = kCoefficients[static_cast<std::uint32_t>(matrix)];
    const bool full_range = (static_cast<std::uint32_t>(matrix) & 1) != 0;
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
    for (; x + 8 <= width; x += 8) {
        // 16 字节解交织为 8 个 Y 和 U0,V0,U2,V2,...，每对色度只属于相邻两像素。
        const auto packed = vld2_u8(input + x * 2);
        const auto u_pairs = vuzp1_u8(packed.val[1], packed.val[1]);
        const auto v_pairs = vuzp2_u8(packed.val[1], packed.val[1]);
        const auto u = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vzip1_u8(u_pairs, u_pairs))), vdupq_n_s16(128));
        const auto v = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vzip1_u8(v_pairs, v_pairs))), vdupq_n_s16(128));
        auto y = vmovl_u8(packed.val[0]);
        if (!full_range) {
            y = vqsubq_u16(y, vdupq_n_u16(16));
        }
        const auto signed_y = vreinterpretq_s16_u16(y);
        const auto lower = color_four(vget_low_s16(signed_y), vget_low_s16(u), vget_low_s16(v), coefficients);
        const auto upper = color_four(vget_high_s16(signed_y), vget_high_s16(u), vget_high_s16(v), coefficients);
        vst1q_u16(output + x, rgb565(clip_eight(lower.red, upper.red), clip_eight(lower.green, upper.green),
                                    clip_eight(lower.blue, upper.blue)));
    }
#endif
    for (; x < width; x += 2) {
        const auto u = int(input[x * 2 + 1]) - 128;
        const auto v = int(input[x * 2 + 3]) - 128;
        const auto red = coefficients[1] * v;
        const auto green = coefficients[2] * u + coefficients[3] * v;
        const auto blue = coefficients[4] * u;
        for (std::uint32_t pixel = 0; pixel < 2; ++pixel) {
            const auto y = full_range ? int(input[x * 2 + pixel * 2]) :
                           std::max(0, int(input[x * 2 + pixel * 2]) - 16);
            const auto luma = y * coefficients[0];
            output[x + pixel] = rgb565(clip_q12(luma + red), clip_q12(luma + green), clip_q12(luma + blue));
        }
    }
}

void put_pixel(std::uint8_t* image, std::uint32_t x, std::uint32_t y, std::uint16_t color) {
    auto* destination = image + y * kDisplayStride + x * 2;
    destination[0] = static_cast<std::uint8_t>(color);
    destination[1] = static_cast<std::uint8_t>(color >> 8);
}

void rectangle(std::uint8_t* image, const Detection& box, const Letterbox& geometry) {
    const auto x1 = std::clamp(int(std::floor(box.x1 * geometry.gain)) + int(geometry.left), 0, 1279);
    const auto y1 = std::clamp(int(std::floor(box.y1 * geometry.gain)) + int(geometry.top), 0, 719);
    const auto x2 = std::clamp(int(std::ceil(box.x2 * geometry.gain)) + int(geometry.left) - 1, x1, 1279);
    const auto y2 = std::clamp(int(std::ceil(box.y2 * geometry.gain)) + int(geometry.top) - 1, y1, 719);
    const auto color = kPalette[box.class_id % 8];
    for (int inset = 0; inset < 2 && x1 + inset <= x2 - inset && y1 + inset <= y2 - inset; ++inset) {
        for (int x = x1 + inset; x <= x2 - inset; ++x) {
            put_pixel(image, x, y1 + inset, color);
            put_pixel(image, x, y2 - inset, color);
        }
        for (int y = y1 + inset; y <= y2 - inset; ++y) {
            put_pixel(image, x1 + inset, y, color);
            put_pixel(image, x2 - inset, y, color);
        }
    }
}

bool same_geometry(const Letterbox& first, const Letterbox& second) {
    return first.source_width == second.source_width && first.source_height == second.source_height &&
           first.target_width == second.target_width && first.target_height == second.target_height &&
           first.resized_width == second.resized_width && first.resized_height == second.resized_height &&
           first.left == second.left && first.top == second.top && first.gain == second.gain;
}

} // namespace

DetectionBundle::DetectionBundle(FrameRef frame, FrameKey inference_key, const Letterbox& model_geometry,
                                 ColorMatrix matrix, std::vector<Detection> model_boxes,
                                 std::uint64_t completed_ns)
    : frame_(std::move(frame)), matrix_(matrix), completed_ns_(completed_ns) {
    if (!frame_ || !(frame_->info().key == inference_key) || completed_ns < frame_->info().captured_ns ||
        static_cast<std::uint32_t>(matrix) > 3 || model_boxes.size() > 300 ||
        (frame_->info().format != PixelFormat::Yuyv && frame_->info().format != PixelFormat::Rgb888)) {
        throw std::invalid_argument("detection identity, time, format or count is invalid");
    }
    const ModelGeometry target{model_geometry.target_width, model_geometry.target_height};
    if (!target.valid()) {
        throw std::invalid_argument("detection model geometry is invalid");
    }
    const auto canonical = make_letterbox(frame_->info().width, frame_->info().height, target.width, target.height);
    if (!same_geometry(canonical, model_geometry)) {
        throw std::invalid_argument("detection transform does not describe the retained source frame");
    }
    boxes_.reserve(model_boxes.size());
    for (auto box : model_boxes) {
        if (!std::isfinite(box.x1) || !std::isfinite(box.x2) || !std::isfinite(box.y1) || !std::isfinite(box.y2) ||
            !std::isfinite(box.score) || box.score < 0 || box.score > 1 || box.class_id >= kClasses ||
            box.x1 > box.x2 || box.y1 > box.y2) {
            throw std::invalid_argument("non-finite or invalid detection cannot be published");
        }
        // 裁剪后退化框可正常丢弃；不以此隐藏非有限值或非法类别。
        if (restore_box(box, canonical)) {
            boxes_.push_back(box);
        }
    }
}

void Compositor720p::configure(const FrameInfo& source) {
    if (source_width_ == source.width && source_height_ == source.height) {
        return;
    }
    geometry_ = make_letterbox(source.width, source.height, 1280, 720);
    for (std::uint32_t x = 0; x < geometry_.resized_width; ++x) {
        horizontal_[x] = std::min(source.width - 1,
                                  (2 * x + 1) * source.width / (2 * geometry_.resized_width));
    }
    for (std::uint32_t y = 0; y < geometry_.resized_height; ++y) {
        vertical_[y] = std::min(source.height - 1,
                                (2 * y + 1) * source.height / (2 * geometry_.resized_height));
    }
    source_width_ = source.width;
    source_height_ = source.height;
}

FrameRef Compositor720p::compose(const DetectionBundle& detection, FramePool& display_pool) {
    if (!detection.frame()) {
        throw std::invalid_argument("cannot compose a moved-from detection bundle");
    }
    auto destination = display_pool.try_acquire();
    if (!destination) {
        return {};
    }
    if (destination->capacity() < kBytes) {
        throw std::invalid_argument("720p display pool slots are too small");
    }
    const auto& source = detection.frame()->info();
    configure(source);
    auto* image = destination->data();
    std::memset(image, 0, kBytes);
    for (std::uint32_t y = 0; y < geometry_.resized_height; ++y) {
        auto* row = image + (y + geometry_.top) * kDisplayStride + geometry_.left * 2;
        if (y && vertical_[y] == vertical_[y - 1]) {
            std::memcpy(row, row - kDisplayStride, geometry_.resized_width * 2);
            continue;
        }
        decode_row(detection.frame()->data() + std::size_t(vertical_[y]) * source.stride, source.width,
                    source.format, detection.matrix(), source_row_.data());
        for (std::uint32_t x = 0; x < geometry_.resized_width; ++x) {
            const auto value = source_row_[horizontal_[x]];
            row[x * 2] = static_cast<std::uint8_t>(value);
            row[x * 2 + 1] = static_cast<std::uint8_t>(value >> 8);
        }
    }
    for (const auto& box : detection.boxes()) {
        rectangle(image, box, geometry_);
    }
    return destination->publish({source.key, source.captured_ns, 1280, 720, kDisplayStride, kBytes,
                                 PixelFormat::Rgb565});
}

} // namespace yolo26
