#include "yolo26/compose.hpp"
#include "yolo26/postprocess.hpp"
#include "yolo26/preprocess.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
volatile std::uint64_t checksum = 0;

template<class Operation>
void measure(const char* name, unsigned iterations, Operation operation) {
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        operation();
    }
    std::vector<std::uint64_t> durations(iterations);
    std::uint64_t total = 0;
    for (auto& duration : durations) {
        const auto begin = Clock::now();
        operation();
        duration = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
        total += duration;
    }
    std::sort(durations.begin(), durations.end());
    const auto rank = [&](std::size_t percent) { return durations[(durations.size() * percent + 99) / 100 - 1]; };
    std::cout << "{\"kind\":\"cpu_module\",\"module\":\"" << name << "\",\"iterations\":" << iterations
              << ",\"warmup\":16,\"mean_ns\":" << total / iterations << ",\"p50_ns\":" << rank(50)
              << ",\"p95_ns\":" << rank(95) << ",\"p99_ns\":" << rank(99)
              << ",\"maximum_ns\":" << durations.back() << ",\"npu_executed\":false}\n";
}

} // namespace

int main(int argc, char** argv) {
    // 显式有界纯 CPU 任务；不打开摄像头、SDK、DMA、显示寄存器，也不注册为自动 CTest。
    if (argc != 2) {
        std::cerr << "usage: yolo26_cpu_benchmark ITERATIONS(8..2000)\n";
        return 2;
    }
    try {
        unsigned iterations = 0;
        const std::string argument(argv[1]);
        const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), iterations);
        if (parsed.ec != std::errc{} || parsed.ptr != argument.data() + argument.size() ||
            iterations < 8 || iterations > 2000) {
            throw std::invalid_argument("invalid finite benchmark iteration count");
        }
        using namespace yolo26;
        std::vector<float> tensor(kRawElements, -10.0F);
        std::fill(tensor.begin(), tensor.begin() + 4 * kAnchors, 1.0F);
        for (std::size_t anchor = 0; anchor < 8; ++anchor) {
            tensor[(4 + anchor % kClasses) * kAnchors + anchor] = 5.0F;
        }
        measure("one_to_one_tail", iterations, [&] {
            const auto boxes = decode_one_to_one(tensor.data(), tensor.size());
            checksum += boxes.size();
        });
        std::vector<std::uint8_t> rgb(640 * 640 * 3, 114);
        std::vector<float> normalized(rgb.size());
        measure("rgb888_to_nhwc_fp32", iterations, [&] {
            normalize_model_input(rgb.data(), rgb.size(), 640 * 3, normalized.data(), normalized.size());
            checksum += static_cast<std::uint64_t>(normalized[128] * 255);
        });
        std::vector<std::uint8_t> packed_rgb(rgb.size());
        measure("rgb888_to_nhwc_u8", iterations, [&] {
            pack_model_input_rgb(rgb.data(), rgb.size(), 640 * 3, packed_rgb.data(), packed_rgb.size());
            checksum += packed_rgb[128];
        });

        FramePool source_pool(1, 640 * 480 * 2);
        FramePool display_pool(3, Compositor720p::kBytes);
        auto writer = source_pool.try_acquire();
        for (std::size_t index = 0; index < writer->capacity(); index += 4) {
            writer->data()[index] = 81;
            writer->data()[index + 1] = 90;
            writer->data()[index + 2] = 145;
            writer->data()[index + 3] = 240;
        }
        auto frame = writer->publish({{1, 1}, 1, 640, 480, 1280, 640 * 480 * 2, PixelFormat::Yuyv});
        DetectionBundle bundle(frame, frame->info().key, make_letterbox(640, 480), ColorMatrix::Bt601Limited,
                                 {{20, 110, 320, 410, 0.9F, 0}, {400, 120, 600, 480, 0.8F, 2}}, 2);
        Compositor720p compositor;
        measure("yuyv_vga_to_720p_with_boxes", iterations, [&] {
            const auto display = compositor.compose(bundle, display_pool);
            if (!display) {
                throw std::runtime_error("benchmark display lease was not returned");
            }
            checksum += display->data()[128 * 2560 + 1000];
        });
        const auto display = compositor.compose(bundle, display_pool);
        std::vector<std::uint8_t> copied(Compositor720p::kBytes);
        measure("host_rgb565_copy", iterations, [&] {
            std::memcpy(copied.data(), display->data(), copied.size());
            checksum += copied[128 * 2560 + 1000];
        });
        std::cout << "{\"kind\":\"summary\",\"input\":\"repeated_synthetic_fixture\",\"checksum\":"
                  << checksum << ",\"postprocess_scan\":\"" << postprocess_scan_backend()
                  << "\",\"normalization\":\"" << normalization_backend()
                  << "\",\"sdk_starts\":0,\"camera_starts\":0,\"npu_executed\":false}\n";
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "CPU benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
