#pragma once
#include "yolo26/frames.hpp"
#include "yolo26/postprocess.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolo26 {

inline constexpr std::size_t kModelRowBytes = kSquareModel.rgb_row_bytes();
inline constexpr std::size_t kModelElements = kSquareModel.rgb_elements();

enum class ColorMatrix : std::uint32_t { Bt601Limited, Bt601Full, Bt709Limited, Bt709Full };

// 此处只描述已由公开分配接口取得的总线地址范围，不代表所有权，也不是 CPU 指针。
// 调用方必须持有真实分配租约，完成缓存同步，并在联合 drained 证明前保留内存。
struct DmaRange {
    std::uint64_t address;
    std::uint64_t capacity;
};

struct PreprocessPlan {
    DmaRange source;
    DmaRange destination;
    std::uint32_t source_stride;
    std::uint32_t destination_stride;
    PixelFormat source_format;
    ColorMatrix matrix;
    Letterbox geometry;
    std::vector<std::uint32_t> horizontal_map;
    std::vector<std::uint32_t> vertical_map;
};

// 生成 half-pixel Q12 采样表，封装与 RTL 相同的 24bit 表项；不使用浮点累加坐标。
std::vector<std::uint32_t> make_axis_map(std::uint32_t source_size, std::uint32_t resized_size);

// 检查 8 字节对齐、二维完整分配、32bit 地址溢出、源目标重叠及硬件支持的几何/格式。
// MJPEG 必须先解码成 RGB。所有检查完成后才返回可用于寄存器编程的完整配置。
PreprocessPlan make_preprocess_plan(DmaRange source, std::uint32_t source_stride,
                                    std::uint32_t width, std::uint32_t height, PixelFormat format,
                                    ColorMatrix matrix, DmaRange destination,
                                    std::uint32_t destination_stride = kModelRowBytes,
                                    ModelGeometry target = kSquareModel);

// 已按模型尺寸整形的 RGB888 -> NHWC FP32 [0,1]，只归一化一次。
// 不允许输入输出区间重叠；非法长度/步长在写入输出前拒绝。缓存同步由适配层先完成。
void normalize_model_input(const std::uint8_t* rgb, std::size_t bytes, std::size_t stride,
                           float* output, std::size_t elements, ModelGeometry geometry = kSquareModel);

// UInt8 输入图使用原始 RGB 字节，由图内完成 /255；这里只去掉行填充，不做数值转换。
// 输出必须恰好为模型 RGB 字节数，且与输入范围不重叠；所有检查先于写入。
void pack_model_input_rgb(const std::uint8_t* rgb, std::size_t bytes, std::size_t stride,
                          std::uint8_t* output, std::size_t output_bytes, ModelGeometry geometry = kSquareModel);

// 仅报告实际编译的数据通路，便于性能记录区分查表与参考除法。
const char* normalization_backend() noexcept;

} // namespace yolo26
