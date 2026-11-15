#include "yolo26/postprocess.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// 原始 FP32 fixture -> 6 个 FP32 字段的检测记录。仅测后处理，不代表整机帧率。
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: postprocess_fixture raw.f32 detections.f32\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != std::streampos(yolo26::kRawElements * sizeof(float))) {
            throw std::runtime_error("unexpected raw fixture size");
        }
        input.seekg(0);
        std::vector<float> raw(yolo26::kRawElements);
        input.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));
        if (!input) {
            throw std::runtime_error("incomplete fixture read");
        }
        std::vector<yolo26::Detection> detections;
        const auto start = std::chrono::steady_clock::now();
        constexpr int repetitions = 50;
        for (int i = 0; i < repetitions; ++i) {
            detections = yolo26::decode_one_to_one(raw.data(), raw.size(), 0.0F);
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        for (const auto& detection : detections) {
            const float row[] = {detection.x1, detection.y1, detection.x2, detection.y2,
                                 detection.score, static_cast<float>(detection.class_id)};
            output.write(reinterpret_cast<const char*>(row), sizeof(row));
        }
        output.close();
        if (!output) {
            throw std::runtime_error("output write failed");
        }
        std::cout << "{\"repetitions\":" << repetitions << ",\"mean_ms\":" << seconds * 1000 / repetitions
                  << ",\"detections\":" << detections.size() << ",\"npu_executed\":false,\"scan_backend\":\""
                  << yolo26::postprocess_scan_backend() << "\"}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
