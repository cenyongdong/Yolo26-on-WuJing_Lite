#pragma once
#include "yolo26/model_geometry.hpp"
#include <cstddef>

namespace icraft::xir { class TensorType; }
namespace icraft::xrt { class Tensor; }

namespace yolo26::detail {

// 借用已有的 FP32 对象。工作区必须比 Tensor 及其副本长寿，且传输期间不能改写。
icraft::xrt::Tensor wrap_host_fp32_input(const icraft::xir::TensorType& type, float* storage,
                                        std::size_t count, ModelGeometry geometry);

// 只覆盖已核验几何的公开无填充 FP32 输出；不猜测设备地址。
void validate_plain_fp32_output(const icraft::xir::TensorType& type, ModelGeometry geometry = kSquareModel);
// 调用者先完成原生 wait、layerCount 核验，并持有设备与目标直到公开 read 返回。
void read_plain_fp32_output(const icraft::xrt::Tensor& tensor, float* destination, std::size_t count,
                            ModelGeometry geometry = kSquareModel);

} // namespace yolo26::detail
