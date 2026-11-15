#include "yolo26/capture.hpp"
#include <linux/videodev2.h>
#include <stdexcept>

namespace yolo26 {

CaptureColor resolve_capture_color(const CaptureFormat& format, ColorMatrix missing_metadata_fallback) {
    if (static_cast<unsigned>(missing_metadata_fallback) > static_cast<unsigned>(ColorMatrix::Bt709Full)) {
        throw std::invalid_argument("unsupported fallback color matrix");
    }
    // 未声明有效的 extended fields 不参与决策；其内存内容可能没有任何格式含义。
    auto encoding = format.extended_fields_valid ? format.ycbcr_encoding : std::uint32_t(V4L2_YCBCR_ENC_DEFAULT);
    auto quantization = format.extended_fields_valid ? format.quantization : std::uint32_t(V4L2_QUANTIZATION_DEFAULT);
    const auto transfer = format.extended_fields_valid ? format.transfer_function : std::uint32_t(V4L2_XFER_FUNC_DEFAULT);
    if (transfer != V4L2_XFER_FUNC_DEFAULT && transfer != V4L2_XFER_FUNC_709 && transfer != V4L2_XFER_FUNC_SRGB) {
        throw std::invalid_argument("camera transfer function requires an unsupported color transform");
    }
    const bool missing = format.colorspace == V4L2_COLORSPACE_DEFAULT;
    if (!missing && format.colorspace != V4L2_COLORSPACE_SMPTE170M && format.colorspace != V4L2_COLORSPACE_REC709 &&
        format.colorspace != V4L2_COLORSPACE_SRGB && format.colorspace != V4L2_COLORSPACE_JPEG &&
        format.colorspace != V4L2_COLORSPACE_470_SYSTEM_M && format.colorspace != V4L2_COLORSPACE_470_SYSTEM_BG) {
        throw std::invalid_argument("camera colorspace is outside supported SDR YUYV conversion");
    }
    if (encoding == V4L2_YCBCR_ENC_DEFAULT) {
        if (missing) {
            encoding = static_cast<unsigned>(missing_metadata_fallback) >= 2 ?
                       V4L2_YCBCR_ENC_709 : V4L2_YCBCR_ENC_601;
        } else {
            encoding = V4L2_MAP_YCBCR_ENC_DEFAULT(format.colorspace);
        }
    }
    if (quantization == V4L2_QUANTIZATION_DEFAULT) {
        if (missing) {
            quantization = (static_cast<unsigned>(missing_metadata_fallback) & 1U) ?
                           V4L2_QUANTIZATION_FULL_RANGE : V4L2_QUANTIZATION_LIM_RANGE;
        } else {
            quantization = V4L2_MAP_QUANTIZATION_DEFAULT(false, format.colorspace, encoding);
        }
    }
    if ((encoding != V4L2_YCBCR_ENC_601 && encoding != V4L2_YCBCR_ENC_709) ||
        (quantization != V4L2_QUANTIZATION_FULL_RANGE && quantization != V4L2_QUANTIZATION_LIM_RANGE)) {
        throw std::invalid_argument("camera YCbCr encoding or quantization is unsupported by PL");
    }
    const unsigned matrix = (encoding == V4L2_YCBCR_ENC_709 ? 2U : 0U) |
                            (quantization == V4L2_QUANTIZATION_FULL_RANGE ? 1U : 0U);
    return {static_cast<ColorMatrix>(matrix), missing, encoding, quantization, transfer};
}

} // namespace yolo26
