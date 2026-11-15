#include "yolo26/compose.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// 生成完整可复核 RGB565 输出。固定伪随机图、四种色彩矩阵、非整向量行尾与行间 padding。
// 本工具只使用 Host 缓冲，不打开设备；结果供优化前后逐字节比较，而非精度或实屏验收。
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: compose_fixture NEW_OUTPUT.rgb565\n";
        return 2;
    }
    try {
        if (std::filesystem::exists(argv[1])) {
            throw std::invalid_argument("preserve previous complete compose output");
        }
        std::ofstream output(argv[1], std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create compose output");
        }
        using namespace yolo26;
        struct Case {
            unsigned width, height;
            PixelFormat format;
            ColorMatrix matrix;
        };
        const std::array<Case, 6> cases{{
            {640, 480, PixelFormat::Yuyv, ColorMatrix::Bt601Limited},
            {640, 480, PixelFormat::Yuyv, ColorMatrix::Bt601Full},
            {640, 480, PixelFormat::Yuyv, ColorMatrix::Bt709Limited},
            {640, 480, PixelFormat::Yuyv, ColorMatrix::Bt709Full},
            {10, 6, PixelFormat::Yuyv, ColorMatrix::Bt601Limited},
            {13, 7, PixelFormat::Rgb888, ColorMatrix::Bt601Limited}}};
        FramePool display_pool(1, Compositor720p::kBytes);
        Compositor720p compositor;
        std::uint64_t sequence = 0;
        for (const auto& item : cases) {
            const auto stride = item.width * (item.format == PixelFormat::Yuyv ? 2U : 3U) + 6U;
            FramePool source_pool(1, stride * item.height);
            auto writer = source_pool.try_acquire();
            std::uint32_t random = 26;
            for (std::size_t index = 0; index < writer->capacity(); ++index) {
                random = random * 1664525U + 1013904223U;
                writer->data()[index] = static_cast<std::uint8_t>(random >> 24);
            }
            const auto source = writer->publish({{26, ++sequence}, 1, item.width, item.height,
                                                  stride, stride * item.height, item.format});
            DetectionBundle bundle(source, source->info().key, make_letterbox(item.width, item.height),
                                     item.matrix, {{20, 110, 320, 410, 0.9F, 0}}, 2);
            const auto display = compositor.compose(bundle, display_pool);
            if (!display) {
                throw std::runtime_error("compose fixture exhausted its bounded pool");
            }
            output.write(reinterpret_cast<const char*>(display->data()), Compositor720p::kBytes);
        }
        output.close();
        if (!output) {
            throw std::runtime_error("complete compose output write failed");
        }
        std::cout << "{\"cases\":" << cases.size() << ",\"bytes\":" << cases.size() * Compositor720p::kBytes
                  << ",\"npu_executed\":false,\"display_started\":false}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
