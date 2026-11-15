#include "yolo26/preprocess.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
#include <arm_neon.h>
#endif

namespace yolo26 {
namespace {

constexpr std::uint64_t kAddressLimit = std::uint64_t{1} << 32;

std::uint64_t checked_extent(DmaRange allocation, std::uint32_t stride,
                             std::uint32_t row_bytes, std::uint32_t rows) {
    const auto rounded = (std::uint64_t(row_bytes) + 7) & ~std::uint64_t{7};
    const auto extent = std::uint64_t(rows - 1) * stride + rounded;
    if (allocation.address >= kAddressLimit || (allocation.address & 7) || (stride & 7) ||
        stride < rounded || allocation.capacity > kAddressLimit - allocation.address ||
        extent > allocation.capacity) {
        throw std::invalid_argument("DMA image exceeds its aligned owned allocation");
    }
    return extent;
}

bool pointer_ranges_overlap(const void* first, std::size_t first_size,
                             const void* second, std::size_t second_size) {
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (first_size > limit - a || second_size > limit - b) {
        throw std::invalid_argument("invalid host memory extent");
    }
    return a < b + second_size && b < a + first_size;
}

std::size_t checked_rgb_extent(const std::uint8_t* rgb, std::size_t bytes, std::size_t stride,
                               ModelGeometry geometry) {
    const auto row_bytes = geometry.rgb_row_bytes();
    if (!geometry.valid() || !rgb || stride < row_bytes ||
        stride > (std::numeric_limits<std::size_t>::max() - row_bytes) / (geometry.height - 1)) {
        throw std::invalid_argument("invalid RGB model input geometry");
    }
    const auto extent = (geometry.height - 1) * stride + row_bytes;
    if (bytes < extent) {
        throw std::invalid_argument("RGB model input is incomplete");
    }
    return extent;
}

} // namespace

std::vector<std::uint32_t> make_axis_map(std::uint32_t source_size, std::uint32_t resized_size) {
    if (source_size == 0 || source_size > 1280 || resized_size == 0 || resized_size > 640) {
        throw std::invalid_argument("unsupported resize map dimensions");
    }
    std::vector<std::uint32_t> entries;
    entries.reserve(resized_size);
    const std::int64_t denominator = 2 * resized_size;
    for (std::uint32_t index = 0; index < resized_size; ++index) {
        const auto numerator = std::int64_t(2 * index + 1) * source_size - resized_size;
        std::uint32_t lower = 0;
        std::uint32_t fraction = 0;
        if (numerator >= std::int64_t(source_size - 1) * denominator) {
            lower = source_size - 1;
        } else if (numerator > 0) {
            lower = static_cast<std::uint32_t>(numerator / denominator);
            fraction = static_cast<std::uint32_t>(((numerator % denominator) * 4096 + denominator / 2) /
                                                   denominator);
            // 舍入进位必须改为下一源像素的小数 0，不能截成错误的 12bit 权重。
            if (fraction == 4096) {
                ++lower;
                fraction = 0;
            }
        }
        entries.push_back(lower | (fraction << 12));
    }
    return entries;
}

PreprocessPlan make_preprocess_plan(DmaRange source, std::uint32_t source_stride,
                                    std::uint32_t width, std::uint32_t height, PixelFormat format,
                                    ColorMatrix matrix, DmaRange destination,
                                    std::uint32_t destination_stride, ModelGeometry target) {
    if (!target.valid()) {
        throw std::invalid_argument("unsupported PL model geometry");
    }
    const auto geometry = make_letterbox(width, height, target.width, target.height);
    if ((format != PixelFormat::Yuyv && format != PixelFormat::Rgb888) ||
        (format == PixelFormat::Yuyv && (width & 1)) || static_cast<std::uint32_t>(matrix) > 3) {
        throw std::invalid_argument("unsupported PL input format or color matrix");
    }
    const auto source_bytes = width * (format == PixelFormat::Rgb888 ? 3U : 2U);
    const auto source_extent = checked_extent(source, source_stride, source_bytes, height);
    const auto destination_extent = checked_extent(destination, destination_stride, target.rgb_row_bytes(),
                                                   target.height);
    if (source.address < destination.address + destination_extent &&
        destination.address < source.address + source_extent) {
        throw std::invalid_argument("source and destination DMA images overlap");
    }
    return {source, destination, source_stride, destination_stride, format, matrix, geometry,
            make_axis_map(width, geometry.resized_width), make_axis_map(height, geometry.resized_height)};
}

void normalize_model_input(const std::uint8_t* rgb, std::size_t bytes, std::size_t stride,
                           float* output, std::size_t elements, ModelGeometry geometry) {
    if (!geometry.valid() || !output || elements != geometry.rgb_elements()) {
        throw std::invalid_argument("invalid FP32 model output geometry");
    }
    const auto extent = checked_rgb_extent(rgb, bytes, stride, geometry);
    if (pointer_ranges_overlap(rgb, extent, output, elements * sizeof(float))) {
        throw std::invalid_argument("invalid or overlapping normalization buffers");
    }
    const auto row_bytes = geometry.rgb_row_bytes();
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
    // 1/255 分解为两个 FP32 常量。低位乘积校正一次舍入的倒数误差；
    // FMA 执行最后舍入，256 种 UInt8 输入均与 FP32 除法逐位相同。
    const auto reciprocal_hi = vdupq_n_f32(0x1.010102p-8F);
    const auto reciprocal_lo = vdupq_n_f32(-0x1.fdfdfep-33F);
    auto store_normalized = [&](float* destination, uint32x4_t bytes32) {
        const auto values = vcvtq_f32_u32(bytes32);
        vst1q_f32(destination, vfmaq_f32(vmulq_f32(values, reciprocal_lo), values, reciprocal_hi));
    };
#endif
    for (std::size_t row = 0; row < geometry.height; ++row) {
        const auto* source = rgb + row * stride;
        auto* destination = output + row * row_bytes;
        std::size_t index = 0;
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
        for (; index + 16 <= row_bytes; index += 16) {
            const auto bytes16 = vld1q_u8(source + index);
            const auto first = vmovl_u8(vget_low_u8(bytes16));
            const auto second = vmovl_high_u8(bytes16);
            store_normalized(destination + index, vmovl_u16(vget_low_u16(first)));
            store_normalized(destination + index + 4, vmovl_high_u16(first));
            store_normalized(destination + index + 8, vmovl_u16(vget_low_u16(second)));
            store_normalized(destination + index + 12, vmovl_high_u16(second));
        }
#endif
        for (; index < row_bytes; ++index) {
            destination[index] = static_cast<float>(source[index]) / 255.0F;
        }
    }
}

void pack_model_input_rgb(const std::uint8_t* rgb, std::size_t bytes, std::size_t stride,
                          std::uint8_t* output, std::size_t output_bytes, ModelGeometry geometry) {
    if (!geometry.valid() || !output || output_bytes != geometry.rgb_elements()) {
        throw std::invalid_argument("invalid UInt8 model output geometry");
    }
    const auto extent = checked_rgb_extent(rgb, bytes, stride, geometry);
    if (pointer_ranges_overlap(rgb, extent, output, output_bytes)) {
        throw std::invalid_argument("overlapping RGB model input buffers");
    }
    const auto row_bytes = geometry.rgb_row_bytes();
    if (stride == row_bytes) {
        std::memcpy(output, rgb, output_bytes);
    } else {
        for (std::size_t row = 0; row < geometry.height; ++row) {
            std::memcpy(output + row * row_bytes, rgb + row * stride, row_bytes);
        }
    }
}

const char* normalization_backend() noexcept {
#if defined(__aarch64__) && !defined(YOLO26_FORCE_SCALAR)
    return "aarch64_neon_fp32_fma";
#else
    return "fp32_division";
#endif
}

} // namespace yolo26
