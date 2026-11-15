#include "yolo26/capture.hpp"
#include <linux/videodev2.h>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class Action> void rejects(Action action) {
    bool failed = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed, "unsupported camera color metadata accepted");
}

} // namespace

int main() {
    using namespace yolo26;
    CaptureFormat format{};
    auto color = resolve_capture_color(format, ColorMatrix::Bt601Limited);
    require(color.assumed_missing_metadata && color.matrix == ColorMatrix::Bt601Limited,
            "missing metadata was not marked as an application assumption");
    format.colorspace = V4L2_COLORSPACE_REC709;
    format.ycbcr_encoding = 999;
    format.quantization = 999;
    format.transfer_function = 999;
    color = resolve_capture_color(format, ColorMatrix::Bt601Limited);
    require(!color.assumed_missing_metadata && color.matrix == ColorMatrix::Bt709Limited,
            "invalid extended fields were consumed instead of colorspace defaults");
    format.extended_fields_valid = true;
    rejects([&] { resolve_capture_color(format, ColorMatrix::Bt601Limited); });
    format.transfer_function = V4L2_XFER_FUNC_DEFAULT;
    format.ycbcr_encoding = V4L2_YCBCR_ENC_601;
    format.quantization = V4L2_QUANTIZATION_FULL_RANGE;
    require(resolve_capture_color(format, ColorMatrix::Bt709Limited).matrix == ColorMatrix::Bt601Full,
            "explicit YCbCr encoding/quantization did not override colorspace defaults");
    format.colorspace = V4L2_COLORSPACE_JPEG;
    format.ycbcr_encoding = V4L2_YCBCR_ENC_DEFAULT;
    format.quantization = V4L2_QUANTIZATION_DEFAULT;
    require(resolve_capture_color(format, ColorMatrix::Bt601Limited).matrix == ColorMatrix::Bt601Full,
            "JPEG default was not full-range BT.601");
    format.colorspace = V4L2_COLORSPACE_BT2020;
    rejects([&] { resolve_capture_color(format, ColorMatrix::Bt601Limited); });
    format.colorspace = V4L2_COLORSPACE_SRGB;
    format.ycbcr_encoding = V4L2_YCBCR_ENC_XV709;
    rejects([&] { resolve_capture_color(format, ColorMatrix::Bt601Limited); });
    format.ycbcr_encoding = V4L2_YCBCR_ENC_709;
    format.transfer_function = V4L2_XFER_FUNC_NONE;
    rejects([&] { resolve_capture_color(format, ColorMatrix::Bt601Limited); });
    std::cout << "camera color defaults, overrides and unsupported conversions passed\n";
}
