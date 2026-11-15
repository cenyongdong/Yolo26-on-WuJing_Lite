#include "icraft_tensor_io.hpp"
#include "yolo26/postprocess.hpp"
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace yolo26::detail {

icraft::xrt::Tensor wrap_host_fp32_input(const icraft::xir::TensorType& type, float* storage,
                                        std::size_t count, ModelGeometry geometry) {
    namespace xir = icraft::xir;
    namespace xrt = icraft::xrt;
    const std::array<std::int64_t, 4> shape{1, geometry.height, geometry.width, 3};
    if (!geometry.valid() || !storage || count != geometry.rgb_elements() ||
        !type.defined() || type->shape.size() != shape.size() ||
        !std::equal(shape.begin(), shape.end(), type->shape.begin()) ||
        !type->element_dtype.isFP32() || type->layout.toString() != xir::Layout("NHWC").toString() ||
        type.bytes() != count * sizeof(float)) {
        throw std::runtime_error("host input requires a complete plain FP32 NHWC buffer");
    }
    const auto& host = xrt::HostDevice::MemRegion();
    // 工作区由运行时拥有，SDK 只释放引用。CPTR 由应用提供，无需转换设备地址。
    auto chunk = host.malloc(xrt::MemPtr(storage), count * sizeof(float), [](const xrt::MemPtr&) {}, true);
    xrt::Tensor input(type, chunk);
    const auto address = input.data();
    if (!input.isReady() || !input.isOn(host) || address.ptype() != xrt::PtrType::CPTR ||
        address.cptr() != reinterpret_cast<char*>(storage)) {
        throw std::runtime_error("host Tensor does not cover the supplied buffer");
    }
    return input;
}

void validate_plain_fp32_output(const icraft::xir::TensorType& type, ModelGeometry geometry) {
    namespace xir = icraft::xir;
    const auto anchors = geometry.anchors();
    const std::array<std::int64_t, 3> shape{1, kRawChannels, static_cast<std::int64_t>(anchors)};
    if (!geometry.valid() || !type.defined() || type->shape.size() != shape.size() ||
        !std::equal(shape.begin(), shape.end(), type->shape.begin()) ||
        !type->element_dtype.is<xir::FloatType>() || !type->element_dtype.isFP32() ||
        type->layout.toString() != xir::Layout("**C").toString() ||
        type.bytes() != geometry.raw_elements() * sizeof(float)) {
        throw std::runtime_error("direct output requires complete plain FP32 software layout");
    }
    if (type->merged_distrs.size() > 1) {
        throw std::runtime_error("direct output has unexpected merged dimensions");
    }
    for (const auto& distribution : type->merged_distrs) {
        if (!distribution.defined() || distribution->merged_axis.size() != 1 ||
            distribution->merged_axis[0] != 2 || distribution.totalNum() != anchors ||
            distribution.totalValidNum() != anchors) {
            throw std::runtime_error("direct output cannot include padded or missing elements");
        }
    }
}

void read_plain_fp32_output(const icraft::xrt::Tensor& tensor, float* destination, std::size_t count,
                            ModelGeometry geometry) {
    if (!geometry.valid() || !tensor.defined() || !tensor.hasData() || !tensor.isReady() || !destination ||
        count != geometry.raw_elements()) {
        throw std::runtime_error("direct output requires a completed tensor and exact destination size");
    }
    validate_plain_fp32_output(tensor.dtype(), geometry);
    // 使用 Tensor::read 的设备传输语义，不把 ADDR/BOTH 或 data() 强转为 Linux 指针。
    tensor.read(reinterpret_cast<char*>(destination), 0, count * sizeof(float));
}

} // namespace yolo26::detail
