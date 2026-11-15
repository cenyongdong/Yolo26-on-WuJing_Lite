#pragma once
#include "yolo26/capture.hpp"
#include "yolo26/icraft_runtime.hpp"
#include "yolo26/metrics.hpp"
#include "yolo26/video_engine.hpp"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace yolo26 {

struct LiveConfig {
    CaptureConfig capture;
    std::uint32_t seconds = 10, warmup_seconds = 2;
    ColorMatrix missing_color_fallback = ColorMatrix::Bt601Limited;
    float threshold = 0.25F;
};

struct LiveFrameTiming {
    FrameKey key;
    std::uint64_t captured_ns, started_ns, completed_ns;
    InferenceTimes stages;
};

struct LiveFrameEvidence {
    FrameInfo source;
    FrameInfo model_input;
    Letterbox transform;
    std::uint64_t inference_completed_ns = 0, adopted_ns = 0;
    DisplaySnapshot adoption{};
    std::vector<std::uint8_t> source_pixels, model_rgb, display_rgb565, adopted_rgb565;
    std::vector<Detection> model_boxes, source_boxes;
    std::vector<char> raw_output;
};

struct LiveReport {
    CaptureFormat format;
    CaptureColor color;
    CaptureCounters camera;
    MetricsReport metrics;
    std::vector<LiveFrameTiming> inference_times;
    std::optional<LiveFrameEvidence> reference;
    std::optional<FrameInfo> failed_input;
    std::vector<std::uint8_t> failure_rgb;
    std::vector<char> failure_output;
    std::uint64_t timing_records_omitted = 0;
    bool camera_closed = false, video_drained = false, interrupted = false;
    std::string camera_error, composition_error, pipeline_error, drain_error;
};

// 调用方已拥有共享板锁，并在当前线程完成 IcraftRuntime::initialize；此函数不再次打开 SDK。
// 只在含 YOLO/720p ABI 的合格位流上调用。它显式 enable_video，运行 1..600 秒测量并停止视频。
// 相机和合成各有一个 CPU 所有者线程；所有 SDK/PL/HDMI 接口仍由调用线程独占。
// 在线只更新有界内存；返回后写报告，慢磁盘不进入采集/推理/显示路径。
// 返回并不隐含 SDK 已关闭。调用方须检查 video_drained，再调用 runtime.close 并保存实际结果。
LiveReport run_live_pipeline(IcraftRuntime& runtime, const LiveConfig& config,
                             const std::function<bool()>& stop_requested);

} // namespace yolo26
