#include "yolo26/metrics.hpp"
#include <algorithm>
#include <limits>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace yolo26 {
namespace {

constexpr std::uint64_t kSecond = 1000000000;

Quantiles quantiles(std::vector<std::uint64_t>& samples) {
    std::sort(samples.begin(), samples.end());
    // nearest-rank，不插值纳秒；调用方保证至少一个样本。
    const auto rank = [&](std::size_t percent) { return samples[(samples.size() * percent + 99) / 100 - 1]; };
    return {rank(50), rank(95), rank(99), samples.back()};
}

void add_checked(std::uint64_t& target, std::uint64_t count) {
    if (count > std::numeric_limits<std::uint64_t>::max() - target) {
        throw std::overflow_error("metrics counter overflow");
    }
    target += count;
}

} // namespace

RunMetrics::RunMetrics(std::uint64_t epoch, std::uint64_t start_ns, std::uint32_t seconds, InputOrigin origin)
    : epoch_(epoch), start_ns_(start_ns), end_ns_(0), seconds_(seconds), origin_(origin), sample_capacity_(0) {
    if (!epoch || !start_ns || !seconds || seconds > 3600 ||
        start_ns > std::numeric_limits<std::uint64_t>::max() - std::uint64_t(seconds) * kSecond ||
        (origin != InputOrigin::LiveCamera && origin != InputOrigin::FileReplay && origin != InputOrigin::Synthetic)) {
        throw std::invalid_argument("invalid bounded measurement epoch or duration");
    }
    end_ns_ = start_ns + std::uint64_t(seconds) * kSecond;
    sample_capacity_ = std::size_t(seconds) * 120;
    for (auto& stage : stages_) {
        stage.summary.windows.resize(seconds, 0);
        stage.durations.reserve(sample_capacity_);
        stage.latencies.reserve(sample_capacity_);
    }
}

EventDisposition RunMetrics::note(MetricStage stage, const StageEvent& event) {
    const auto index = static_cast<std::size_t>(stage);
    if (index >= stages_.size()) {
        throw std::invalid_argument("unknown metric stage");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        throw std::logic_error("measurement is finalized");
    }
    auto& state = stages_[index];
    auto& summary = state.summary;
    if (event.key.epoch != epoch_) {
        ++summary.wrong_epoch;
        return EventDisposition::WrongEpoch;
    }
    if (state.have_identity && event.key.sequence <= state.last_sequence) {
        ++summary.duplicate_or_reordered;
        return EventDisposition::DuplicateOrReordered;
    }
    if (!event.captured_ns || event.started_ns < event.captured_ns || event.completed_ns < event.started_ns ||
        (state.have_identity && event.completed_ns < state.last_completion_ns)) {
        ++summary.invalid_time;
        return EventDisposition::InvalidTime;
    }
    state.have_identity = true;
    state.last_sequence = event.key.sequence;
    state.last_completion_ns = event.completed_ns;
    if (event.completed_ns < start_ns_ || event.completed_ns >= end_ns_) {
        ++summary.outside;
        return EventDisposition::OutsideWindow;
    }
    auto& window = summary.windows[(event.completed_ns - start_ns_) / kSecond];
    if (window == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("per-second frame count overflow");
    }
    ++window;
    ++summary.new_frames;
    const auto previous = state.last_counted_ns ? state.last_counted_ns : start_ns_;
    summary.longest_gap_ns = std::max(summary.longest_gap_ns, event.completed_ns - previous);
    state.last_counted_ns = event.completed_ns;
    latest_counted_ns_ = std::max(latest_counted_ns_, event.completed_ns);
    if (state.durations.size() < sample_capacity_) {
        state.durations.push_back(event.completed_ns - event.started_ns);
        state.latencies.push_back(event.completed_ns - event.captured_ns);
    } else {
        ++summary.latency_samples_omitted;
    }
    return EventDisposition::Counted;
}

void RunMetrics::note_drop(DropReason reason, std::uint64_t count) {
    const auto index = static_cast<std::size_t>(reason);
    if (index >= drops_.size()) {
        throw std::invalid_argument("unknown drop reason");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        throw std::logic_error("measurement is finalized");
    }
    add_checked(drops_[index], count);
}

void RunMetrics::note_refreshes(std::uint64_t count, std::uint64_t observed_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        throw std::logic_error("measurement is finalized");
    }
    if (observed_ns >= start_ns_ && observed_ns < end_ns_) {
        add_checked(refreshes_, count);
        latest_counted_ns_ = std::max(latest_counted_ns_, observed_ns);
    }
}

void RunMetrics::observe_pools(std::size_t raw_in_use, std::size_t input_in_use, std::size_t display_in_use) {
    if (raw_in_use > 6 || input_in_use > 3 || display_in_use > 3) {
        throw std::invalid_argument("pool occupancy exceeds fixed pipeline allocation");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        throw std::logic_error("measurement is finalized");
    }
    pool_high_water_[0] = std::max(pool_high_water_[0], raw_in_use);
    pool_high_water_[1] = std::max(pool_high_water_[1], input_in_use);
    pool_high_water_[2] = std::max(pool_high_water_[2], display_in_use);
}

MetricsReport RunMetrics::finalize(std::uint64_t observed_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        throw std::logic_error("measurement already finalized");
    }
    if (observed_ns < start_ns_ || observed_ns < latest_counted_ns_) {
        throw std::invalid_argument("final time precedes measured events");
    }
    const auto actual_end = std::min(observed_ns, end_ns_);
    MetricsReport report{};
    report.origin = origin_;
    report.epoch = epoch_;
    report.start_ns = start_ns_;
    report.end_ns = actual_end;
    report.duration_seconds = seconds_;
    report.measurement_complete = observed_ns >= end_ns_;
    report.drops = drops_;
    report.pool_high_water = pool_high_water_;
    report.hdmi_refreshes = refreshes_;
    for (std::size_t index = 0; index < stages_.size(); ++index) {
        auto& state = stages_[index];
        auto& summary = report.stages[index];
        summary = state.summary;
        const auto tail_start = state.last_counted_ns ? state.last_counted_ns : start_ns_;
        summary.longest_gap_ns = std::max(summary.longest_gap_ns, actual_end - tail_start);
        summary.mean_fps = actual_end > start_ns_ ?
                           double(summary.new_frames) * kSecond / double(actual_end - start_ns_) : 0;
        if (!state.durations.empty() && summary.latency_samples_omitted == 0) {
            summary.stage_duration = quantiles(state.durations);
            summary.capture_to_completion = quantiles(state.latencies);
        }
    }
    const auto& inference = report.stages[static_cast<std::size_t>(MetricStage::Inference)];
    report.every_inference_second_at_least_30 = report.measurement_complete &&
        std::all_of(inference.windows.begin(), inference.windows.end(), [](auto count) { return count >= 30; });
    report.camera_30fps_10min_passed = report.every_inference_second_at_least_30 && seconds_ >= 600 &&
                                      origin_ == InputOrigin::LiveCamera;
    finalized_ = true;
    return report;
}

void write_metrics_json(std::ostream& destination, const MetricsReport& report) {
    // 不继承调用方的 hex/boolalpha/locale 等格式，保证输出始终是合法 JSON。
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(12);
    constexpr const char* names[] = {"capture", "preprocess", "inference", "postprocess", "compose", "display"};
    constexpr const char* drop_names[] = {"capture_pool_full", "latest_replaced", "input_pool_full",
                                         "display_pool_full", "invalid_result", "device_error", "log_queue_full"};
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    const auto distribution = [&](const std::optional<Quantiles>& values) {
        if (!values) {
            output << "null";
        } else {
            output << "{\"p50_ns\":" << values->p50_ns << ",\"p95_ns\":" << values->p95_ns
                   << ",\"p99_ns\":" << values->p99_ns << ",\"maximum_ns\":" << values->maximum_ns << '}';
        }
    };
    const auto origin = report.origin == InputOrigin::LiveCamera ? "live_camera" :
                        report.origin == InputOrigin::FileReplay ? "file_replay" : "synthetic";
    output << "{\"epoch\":" << report.epoch << ",\"origin\":\"" << origin << "\",\"start_ns\":"
           << report.start_ns << ",\"end_ns\":" << report.end_ns << ",\"planned_seconds\":" << report.duration_seconds
           << ",\"measurement_complete\":" << boolean(report.measurement_complete)
           << ",\"every_inference_second_at_least_30\":" << boolean(report.every_inference_second_at_least_30)
           << ",\"camera_30fps_10min_passed\":" << boolean(report.camera_30fps_10min_passed)
           << ",\"hdmi_refreshes\":" << report.hdmi_refreshes << ",\"stages\":{";
    for (std::size_t index = 0; index < report.stages.size(); ++index) {
        const auto& stage = report.stages[index];
        output << (index ? "," : "") << '"' << names[index] << "\":{\"new_frames\":" << stage.new_frames
               << ",\"mean_fps\":" << stage.mean_fps << ",\"outside\":" << stage.outside
               << ",\"duplicate_or_reordered\":" << stage.duplicate_or_reordered
               << ",\"wrong_epoch\":" << stage.wrong_epoch << ",\"invalid_time\":" << stage.invalid_time
               << ",\"longest_gap_ns\":" << stage.longest_gap_ns
               << ",\"latency_samples_omitted\":" << stage.latency_samples_omitted
               << ",\"stage_duration\":";
        distribution(stage.stage_duration);
        output << ",\"capture_to_completion\":";
        distribution(stage.capture_to_completion);
        output << ",\"windows\":[";
        for (std::size_t second = 0; second < stage.windows.size(); ++second) {
            output << (second ? "," : "") << stage.windows[second];
        }
        output << "]}";
    }
    output << "},\"drops\":{";
    for (std::size_t index = 0; index < report.drops.size(); ++index) {
        output << (index ? "," : "") << '"' << drop_names[index] << "\":" << report.drops[index];
    }
    output << "},\"pool_high_water\":{\"raw\":" << report.pool_high_water[0] << ",\"input\":"
           << report.pool_high_water[1] << ",\"display\":" << report.pool_high_water[2] << "}}\n";
    const auto json = output.str();
    destination.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!destination) {
        throw std::runtime_error("metrics report write failed");
    }
}

} // namespace yolo26
