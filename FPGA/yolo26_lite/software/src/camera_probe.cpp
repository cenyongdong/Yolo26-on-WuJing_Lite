#include "yolo26/capture.hpp"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t stopping = 0;
void request_stop(int) {
    stopping = 1;
}

unsigned parse_number(const char* text, unsigned minimum, unsigned maximum) {
    const std::string value(text);
    unsigned result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        result < minimum || result > maximum) {
        throw std::invalid_argument("invalid bounded camera argument");
    }
    return result;
}

void save_frame(const yolo26::FrameRef& frame, const char* path) {
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        throw std::runtime_error("cannot create fresh last-frame file");
    }
    std::size_t written = 0;
    while (written < frame->info().bytes_used) {
        const auto count = ::write(fd, frame->data() + written, frame->info().bytes_used - written);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error("last-frame write failed");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::close(fd) < 0) {
        throw std::runtime_error("last-frame close failed");
    }
}

struct Trace {
    std::uint64_t sequence, captured_ns, received_ns;
};

} // namespace

int main(int argc, char** argv) {
    // 此工具只能经持共享板锁且已核验上一轮排空的控制器显式启动，不注册为 CTest。
    if (argc != 5 && argc != 8) {
        std::cerr << "usage: yolo26_camera_probe DEVICE MEASURE_SECONDS WARMUP_SECONDS NEW_LAST_FRAME_FILE"
                     " [WIDTH HEIGHT FPS]\n";
        return 2;
    }
    try {
        const auto seconds = parse_number(argv[2], 1, 600);
        const auto warmup = parse_number(argv[3], 0, 30);
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        yolo26::CaptureConfig config;
        config.device = argv[1];
        if (argc == 8) {
            config.width = parse_number(argv[5], 1, 1280);
            config.height = parse_number(argv[6], 1, 720);
            config.fps = parse_number(argv[7], 1, 60);
        }
        config.epoch = yolo26::monotonic_ns();
        yolo26::CameraCapture camera(config);
        const auto format = camera.format();
        yolo26::FramePool pool(6, format.size_image);
        std::vector<Trace> traces;
        traces.reserve((seconds + warmup) * 120 + 16);
        std::vector<std::uint32_t> windows(seconds, 0);
        camera.start();
        const auto start = yolo26::monotonic_ns();
        const auto measurement_start = start + std::uint64_t(warmup) * 1000000000;
        const auto deadline = measurement_start + std::uint64_t(seconds) * 1000000000;
        std::uint64_t last_sequence = 0, measured = 0;
        yolo26::FrameRef last_frame;
        while (!stopping && yolo26::monotonic_ns() < deadline) {
            auto frame = camera.next(pool, 100);
            if (!frame) {
                continue;
            }
            const auto observed = yolo26::monotonic_ns();
            if (frame->info().key.epoch != config.epoch || frame->info().key.sequence <= last_sequence) {
                throw std::runtime_error("capture published a duplicate or wrong-epoch frame");
            }
            last_sequence = frame->info().key.sequence;
            if (traces.size() == traces.capacity()) {
                throw std::runtime_error("capture trace bound exceeded");
            }
            traces.push_back({last_sequence, frame->info().captured_ns, observed});
            if (observed >= measurement_start && observed < deadline) {
                ++windows[(observed - measurement_start) / 1000000000];
                ++measured;
            }
            last_frame = std::move(frame);
        }
        const bool clean = camera.stop();
        const auto& stats = camera.counters();
        if (last_frame) {
            save_frame(last_frame, argv[4]);
        }
        last_frame.reset();
        std::cout << "{\"kind\":\"format\",\"epoch\":" << config.epoch
                  << ",\"requested_width\":" << config.width << ",\"requested_height\":" << config.height
                  << ",\"requested_fps\":" << config.fps
                  << ",\"width\":" << format.width << ",\"height\":" << format.height
                  << ",\"stride\":" << format.stride << ",\"size_image\":" << format.size_image
                  << ",\"buffers\":" << format.buffers << ",\"interval_numerator\":" << format.frame_interval_numerator
                  << ",\"interval_denominator\":" << format.frame_interval_denominator
                  << ",\"colorspace\":" << format.colorspace << ",\"ycbcr_encoding\":" << format.ycbcr_encoding
                  << ",\"quantization\":" << format.quantization << ",\"transfer_function\":" << format.transfer_function
                  << ",\"extended_fields_valid\":" << (format.extended_fields_valid ? "true" : "false") << "}\n";
        for (const auto& trace : traces) {
            std::cout << "{\"kind\":\"frame\",\"sequence\":" << trace.sequence
                      << ",\"captured_ns\":" << trace.captured_ns << ",\"received_ns\":" << trace.received_ns << "}\n";
        }
        std::cout << "{\"kind\":\"summary\",\"measurement_start_ns\":" << measurement_start
                  << ",\"measurement_end_ns\":" << deadline << ",\"duration_seconds\":" << seconds
                  << ",\"interrupted\":" << (stopping ? "true" : "false") << ",\"new_frames\":" << measured
                  << ",\"mean_fps\":" << static_cast<double>(measured) / seconds
                  << ",\"windows\":[";
        for (std::size_t index = 0; index < windows.size(); ++index) {
            std::cout << (index ? "," : "") << windows[index];
        }
        std::cout << "],\"dequeued\":" << stats.dequeued << ",\"published\":" << stats.published
                  << ",\"driver_sequence_gaps\":" << stats.driver_sequence_gaps
                  << ",\"duplicates_or_reordered\":" << stats.duplicate_or_reordered
                  << ",\"flagged_error\":" << stats.flagged_error << ",\"invalid_bytes\":" << stats.invalid_bytes
                  << ",\"invalid_timestamp\":" << stats.invalid_timestamp << ",\"latest_replaced\":" << stats.latest_replaced
                  << ",\"pool_exhausted\":" << stats.pool_exhausted << ",\"poll_timeouts\":" << stats.poll_timeouts
                  << ",\"driver_timestamps\":" << stats.driver_timestamps
                  << ",\"receive_timestamps\":" << stats.receive_timestamps
                  << ",\"pool_available\":" << pool.available() << ",\"clean_shutdown\":" << (clean ? "true" : "false") << "}\n";
        return clean && !stopping && measured > 0 && pool.available() == 6 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "camera probe failed: " << error.what() << '\n';
        return 1;
    }
}
