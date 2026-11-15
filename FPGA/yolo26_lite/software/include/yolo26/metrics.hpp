#pragma once
#include "yolo26/frames.hpp"
#include <array>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <optional>
#include <vector>

namespace yolo26 {

enum class MetricStage : std::size_t { Capture, Preprocess, Inference, Postprocess, Compose, Display, Count };
enum class InputOrigin { LiveCamera, FileReplay, Synthetic };
enum class EventDisposition { Counted, OutsideWindow, DuplicateOrReordered, WrongEpoch, InvalidTime };
enum class DropReason : std::size_t { CapturePoolFull, LatestReplaced, InputPoolFull, DisplayPoolFull,
                                     InvalidResult, DeviceError, LogQueueFull, Count };

struct StageEvent {
    FrameKey key;
    std::uint64_t captured_ns;
    std::uint64_t started_ns;
    std::uint64_t completed_ns;
};

struct Quantiles {
    std::uint64_t p50_ns, p95_ns, p99_ns, maximum_ns;
};

struct StageSummary {
    std::uint64_t new_frames = 0, outside = 0, duplicate_or_reordered = 0, wrong_epoch = 0, invalid_time = 0;
    std::vector<std::uint32_t> windows;
    double mean_fps = 0;
    std::uint64_t longest_gap_ns = 0;
    std::uint64_t latency_samples_omitted = 0;
    std::optional<Quantiles> stage_duration, capture_to_completion;
};

struct MetricsReport {
    InputOrigin origin;
    std::uint64_t epoch, start_ns, end_ns;
    std::uint32_t duration_seconds;
    std::array<StageSummary, static_cast<std::size_t>(MetricStage::Count)> stages;
    std::array<std::uint64_t, static_cast<std::size_t>(DropReason::Count)> drops;
    std::array<std::size_t, 3> pool_high_water;
    std::uint64_t hdmi_refreshes;
    bool measurement_complete;
    bool every_inference_second_at_least_30;
    bool camera_30fps_10min_passed;
};

// 在停止视频工作后或独立记录线程调用；事件更新路径不调用此函数。
void write_metrics_json(std::ostream& output, const MetricsReport& report);

// 各生产者在线只更新固定容量内存，不写文件；finalize 在停止提交事件后使用。
// 每阶段由单一执行者按帧序完成，独立检查 epoch/sequence；重复扫描从单独接口计数。
// 事件必须在真实有效完成后提交。这个统计类本身不能证明 SDK 已执行或帧来自实时摄像头。
class RunMetrics {
public:
    RunMetrics(std::uint64_t epoch, std::uint64_t start_ns, std::uint32_t seconds, InputOrigin origin);
    EventDisposition note(MetricStage stage, const StageEvent& event);
    void note_drop(DropReason reason, std::uint64_t count = 1);
    void note_refreshes(std::uint64_t count, std::uint64_t observed_ns);
    void observe_pools(std::size_t raw_in_use, std::size_t input_in_use, std::size_t display_in_use);
    MetricsReport finalize(std::uint64_t observed_ns);

private:
    struct StageState {
        StageSummary summary;
        bool have_identity = false;
        std::uint64_t last_sequence = 0, last_completion_ns = 0, last_counted_ns = 0;
        std::vector<std::uint64_t> durations, latencies;
    };
    std::uint64_t epoch_, start_ns_, end_ns_;
    std::uint32_t seconds_;
    InputOrigin origin_;
    std::size_t sample_capacity_;
    std::array<StageState, static_cast<std::size_t>(MetricStage::Count)> stages_;
    std::array<std::uint64_t, static_cast<std::size_t>(DropReason::Count)> drops_{};
    std::array<std::size_t, 3> pool_high_water_{};
    std::uint64_t refreshes_ = 0, latest_counted_ns_ = 0;
    bool finalized_ = false;
    std::mutex mutex_;
};

} // namespace yolo26
