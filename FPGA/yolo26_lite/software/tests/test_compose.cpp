#include "yolo26/compose.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

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
    require(rejected, "invalid bundle or display accepted");
}

std::uint16_t pixel(const yolo26::FrameRef& frame, unsigned x, unsigned y) {
    const auto* value = frame->data() + y * frame->info().stride + x * 2;
    return std::uint16_t(value[0]) | std::uint16_t(value[1]) << 8;
}

} // namespace

int main() {
    using namespace yolo26;
    FramePool source_pool(1, 640 * 480 * 3 + 480 * 8);
    FramePool display_pool(3, Compositor720p::kBytes);
    Compositor720p compositor;
    auto writer = source_pool.try_acquire();
    constexpr unsigned stride = 640 * 3 + 8;
    std::fill(writer->data(), writer->data() + writer->capacity(), 0xa5);
    for (unsigned y = 0; y < 480; ++y) {
        for (unsigned x = 0; x < 640; ++x) {
            auto* output = writer->data() + y * stride + x * 3;
            output[0] = std::uint8_t(x);
            output[1] = std::uint8_t(y);
            output[2] = 0;
        }
    }
    auto source = writer->publish({{7, 12}, 100, 640, 480, stride, stride * 480, PixelFormat::Rgb888});
    const auto geometry = make_letterbox(640, 480);
    DetectionBundle plain(source, source->info().key, geometry, ColorMatrix::Bt601Limited, {}, 200);
    auto display = compositor.compose(plain, display_pool);
    require(display->info().key == source->info().key && display->info().captured_ns == 100 &&
            display->info().format == PixelFormat::Rgb565, "display lost source identity");
    // 独立按 3:2 显示比例核对完整 720p，包含 160 像素左右黑边及源行 padding。
    for (unsigned y = 0; y < 720; ++y) {
        for (unsigned x = 0; x < 1280; ++x) {
            std::uint16_t expected = 0;
            if (x >= 160 && x < 1120) {
                const auto sx = unsigned(std::floor((x - 160 + 0.5) / 1.5));
                const auto sy = unsigned(std::floor((y + 0.5) / 1.5));
                expected = std::uint16_t((std::uint8_t(sx) >> 3) << 11 | (std::uint8_t(sy) >> 2) << 5);
            }
            require(pixel(display, x, y) == expected, "source pixel, endian or display aspect mismatch");
        }
    }
    DetectionBundle boxes(source, source->info().key, geometry, ColorMatrix::Bt601Limited,
                           {{10, 100, 110, 180, 0.9F, 0}, {-40, -50, -20, -10, 0.8F, 1}}, 201);
    require(boxes.boxes().size() == 1, "off-image degenerate box retained");
    auto drawn = compositor.compose(boxes, display_pool);
    // 原图 (10,20)..(110,100) 映射到显示 (175,30)..(325,150)，右下边界为开区间。
    require(pixel(drawn, 175, 30) == 0x07e0 && pixel(drawn, 324, 149) == 0x07e0 &&
            pixel(drawn, 325, 150) == pixel(display, 325, 150) &&
            pixel(drawn, 200, 80) == pixel(display, 200, 80), "box mapping or interior modified");
    {
        DetectionBundle rectangular(source, source->info().key, make_letterbox(640, 480, 512, 384),
                                    ColorMatrix::Bt601Limited, {{8, 16, 88, 80, 0.9F, 0}}, 201);
        const auto rect = compositor.compose(rectangular, display_pool);
        require(std::equal(rect->data(), rect->data() + Compositor720p::kBytes, drawn->data()),
                "rectangular detection did not map to the same VGA frame and display pixels");
    }
    auto third = compositor.compose(plain, display_pool);
    require(!compositor.compose(plain, display_pool) && display_pool.available() == 0,
            "display pool overwrote an active frame");
    third.reset();
    require(display_pool.available() == 1 && compositor.compose(plain, display_pool), "display lease not reclaimed");
    rejects([&] { DetectionBundle bad(source, {7, 13}, geometry, ColorMatrix::Bt601Limited, {}, 200); });
    rejects([&] { DetectionBundle bad(source, {8, 12}, geometry, ColorMatrix::Bt601Limited, {}, 200); });
    rejects([&] { DetectionBundle bad(source, {7, 12}, geometry, ColorMatrix::Bt601Limited, {}, 99); });
    auto wrong_geometry = geometry;
    ++wrong_geometry.top;
    rejects([&] { DetectionBundle bad(source, {7, 12}, wrong_geometry, ColorMatrix::Bt601Limited, {}, 200); });
    rejects([&] { DetectionBundle bad(source, {7, 12}, geometry, ColorMatrix::Bt601Limited,
                    {{0, 0, 1, 1, std::numeric_limits<float>::quiet_NaN(), 0}}, 200); });
    rejects([&] { DetectionBundle bad(source, {7, 12}, geometry, ColorMatrix::Bt601Limited,
                    {{0, 0, 1, 1, 0.5F, 80}}, 200); });
    FramePool undersized(1, 32);
    rejects([&] { compositor.compose(plain, undersized); });
    require(undersized.available() == 1, "failed compose leaked CPU pool slot");
    source.reset();
    require(source_pool.available() == 0, "bundle failed to retain original image");

    // YUYV 黑白阶梯 + 饱和红色，分别覆盖四种矩阵、有限/全范围与 RGB565 小端。
    constexpr std::uint8_t samples[4][8] = {
        {16, 128, 235, 128, 81, 90, 81, 240}, {0, 128, 255, 128, 76, 85, 76, 255},
        {16, 128, 235, 128, 63, 102, 63, 240}, {0, 128, 255, 128, 54, 99, 54, 255}};
    for (unsigned matrix = 0; matrix < 4; ++matrix) {
        FramePool yuv_pool(1, 8);
        auto yuv_writer = yuv_pool.try_acquire();
        std::copy(samples[matrix], samples[matrix] + 8, yuv_writer->data());
        auto yuv = yuv_writer->publish({{8, matrix + 1}, 300, 4, 1, 8, 8, PixelFormat::Yuyv});
        DetectionBundle yuv_bundle(yuv, yuv->info().key, make_letterbox(4, 1), ColorMatrix(matrix), {}, 400);
        auto yuv_display = compositor.compose(yuv_bundle, display_pool);
        require(pixel(yuv_display, 100, 300) == 0 && pixel(yuv_display, 400, 300) == 0xffff &&
                pixel(yuv_display, 700, 300) == 0xf800, "YUYV color conversion mismatch");
        require(pixel(yuv_display, 400, 0) == 0 && pixel(yuv_display, 400, 719) == 0,
                "wide source letterbox mismatch");
    }
    std::cout << "PASS compose: same-frame ownership, full 720p pixels, boxes, four matrices and bounded pools\n";
}
