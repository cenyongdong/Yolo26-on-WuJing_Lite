#include "yolo26/preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class Action>
void rejects(Action action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid configuration was accepted");
}

} // namespace

int main() {
    using namespace yolo26;
    // 独立以浮点坐标验证离散映射误差，覆盖源宽上限、大小缩放和边界复制。
    for (std::uint32_t source = 1; source <= 1280; ++source) {
        for (const std::uint32_t resized : {1, 17, 320, 640}) {
            const auto entries = make_axis_map(source, resized);
            for (std::size_t index = 0; index < entries.size(); ++index) {
                const auto lower = entries[index] & 2047;
                const auto fraction = entries[index] >> 12;
                const double coordinate = std::clamp((index + 0.5) * source / resized - 0.5,
                                                     0.0, double(source - 1));
                const double actual = lower + fraction / 4096.0;
                require(lower < source && fraction < 4096 && !(entries[index] & 2048), "invalid LUT field");
                require(std::abs(coordinate - actual) <= 0.5 / 4096 + 1e-10, "half-pixel mapping error");
                if (index) {
                    const auto previous = entries[index - 1];
                    require(actual >= (previous & 2047) + (previous >> 12) / 4096.0, "mapping reversed");
                }
            }
        }
    }
    const DmaRange source{0x10000000, 614400};
    const DmaRange destination{0x20000000, 1228800};
    const auto make_plan = [&](DmaRange src, std::uint32_t stride, DmaRange dst) {
        return make_preprocess_plan(src, stride, 640, 480, PixelFormat::Yuyv,
                                    ColorMatrix::Bt601Limited, dst);
    };
    const auto plan = make_plan(source, 1280, destination);
    require(plan.geometry.top == 80 && plan.vertical_map.size() == 480 &&
            plan.horizontal_map.size() == 640, "VGA plan geometry");
    require(plan.horizontal_map[639] == 639 && plan.vertical_map[479] == 479, "identity LUT");
    rejects([&] { make_plan({source.address + 1, source.capacity}, 1280, destination); });
    rejects([&] { make_plan(source, 1272, destination); });
    rejects([&] { make_plan({source.address, source.capacity - 1}, 1280, destination); });
    rejects([&] { make_plan({0xfffffff8, 614400}, 1280, destination); });
    rejects([&] { make_plan(source, 1280, {source.address + 8, destination.capacity}); });
    rejects([&] { make_preprocess_plan(source, 1280, 639, 480, PixelFormat::Yuyv,
                                       ColorMatrix::Bt601Limited, destination); });
    rejects([&] { make_preprocess_plan(source, 1280, 640, 480, PixelFormat::Mjpeg,
                                       ColorMatrix::Bt601Limited, destination); });
    rejects([&] { make_axis_map(1281, 640); });
    rejects([&] { make_axis_map(640, 0); });
    // 总线最后一个可寻址字节恰好在分配末尾时合法，不能因 32bit 加法溢出误拒绝。
    make_plan(source, 1280, {(std::uint64_t{1} << 32) - 1228800, 1228800});

    constexpr std::size_t stride = 1936;
    std::vector<std::uint8_t> image(640 * stride, 0xa5);
    std::vector<float> output(640 * 640 * 3, -1.0F);
    for (std::size_t row = 0; row < 640; ++row) {
        for (std::size_t column = 0; column < 1920; ++column) {
            image[row * stride + column] = static_cast<std::uint8_t>(row + column);
        }
    }
    normalize_model_input(image.data(), image.size(), stride, output.data(), output.size());
    for (std::size_t row = 0; row < 640; ++row) {
        for (std::size_t column = 0; column < 1920; ++column) {
            const float expected = static_cast<float>(image[row * stride + column]) / 255.0F;
            require(std::memcmp(&output[row * 1920 + column], &expected, sizeof(float)) == 0,
                    "NHWC layout, row padding or exact FP32 normalization changed");
        }
    }
    const auto saved = output;
    rejects([&] { normalize_model_input(image.data(), image.size() - 17, stride, output.data(), output.size()); });
    require(output == saved, "invalid normalization partially wrote output");
    rejects([&] { normalize_model_input(image.data(), image.size(), std::numeric_limits<std::size_t>::max(),
                                         output.data(), output.size()); });
    rejects([&] { normalize_model_input(reinterpret_cast<const std::uint8_t*>(output.data()), output.size() * 4,
                                         1920, output.data(), output.size()); });

    std::vector<std::uint8_t> packed(kModelElements, 0x5a), compact(kModelElements);
    pack_model_input_rgb(image.data(), image.size(), stride, packed.data(), packed.size());
    for (std::size_t row = 0; row < 640; ++row) {
        require(std::memcmp(packed.data() + row * kModelRowBytes, image.data() + row * stride,
                            kModelRowBytes) == 0, "UInt8 model input changed RGB or included row padding");
    }
    pack_model_input_rgb(packed.data(), packed.size(), kModelRowBytes, compact.data(), compact.size());
    require(compact == packed, "compact RGB copy changed model input");
    const auto original_bytes = packed;
    rejects([&] { pack_model_input_rgb(image.data(), image.size() - 17, stride, packed.data(), packed.size()); });
    rejects([&] { pack_model_input_rgb(image.data(), image.size(), 1919, packed.data(), packed.size()); });
    rejects([&] { pack_model_input_rgb(image.data(), image.size(), std::numeric_limits<std::size_t>::max(),
                                      packed.data(), packed.size()); });
    rejects([&] { pack_model_input_rgb(image.data(), image.size(), stride, packed.data(), packed.size() - 1); });
    rejects([&] { pack_model_input_rgb(packed.data(), packed.size(), 1920, packed.data(), packed.size()); });
    rejects([&] { pack_model_input_rgb(nullptr, image.size(), stride, packed.data(), packed.size()); });
    rejects([&] { pack_model_input_rgb(image.data(), image.size(), stride, nullptr, packed.size()); });
    require(packed == original_bytes, "invalid RGB packing partially wrote output");
    std::cout << "PASS preprocess plan: LUT bounds, owned DMA extents, overlap and NHWC normalization\n";
}
