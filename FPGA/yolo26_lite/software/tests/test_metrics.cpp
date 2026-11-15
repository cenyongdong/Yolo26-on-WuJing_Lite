#include "yolo26/metrics.hpp"
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr std::uint64_t start = 1000000000, second = 1000000000;

yolo26::StageEvent event(std::uint64_t sequence, std::uint64_t completed) {
    return {{26, sequence}, completed - 20000000, completed - 10000000, completed};
}

// 用合成事件核验统计器，不得将本测试的 30FPS 当作硬件吞吐证据。
yolo26::MetricsReport run_trace(yolo26::InputOrigin origin, bool slow_window) {
    yolo26::RunMetrics metrics(26, start, 600, origin);
    std::uint64_t sequence = 0;
    for (unsigned window = 0; window < 600; ++window) {
        // 59/1 的相邻秒均值仍为 30；严格口径必须拒绝后一秒。
        const auto count = slow_window && window == 100 ? 59U : slow_window && window == 101 ? 1U : 30U;
        for (unsigned frame = 0; frame < count; ++frame) {
            const auto when = start + window * second + (frame + 1) * second / (count + 1);
            require(metrics.note(yolo26::MetricStage::Inference, event(++sequence, when)) ==
                    yolo26::EventDisposition::Counted, "new inference not counted");
            metrics.note_refreshes(2, when);
        }
    }
    return metrics.finalize(start + 600 * second);
}

} // namespace

int main() {
    using namespace yolo26;
    const auto complete = run_trace(InputOrigin::LiveCamera, false);
    const auto inference_index = static_cast<std::size_t>(MetricStage::Inference);
    require(complete.camera_30fps_10min_passed && complete.hdmi_refreshes == 36000 &&
            complete.stages[inference_index].new_frames == 18000, "strict live-camera counting");
    const auto uneven = run_trace(InputOrigin::LiveCamera, true);
    require(uneven.stages[inference_index].mean_fps == 30 && !uneven.camera_30fps_10min_passed,
            "mean FPS hid a slow one-second window");
    const auto replay = run_trace(InputOrigin::FileReplay, false);
    require(replay.every_inference_second_at_least_30 && !replay.camera_30fps_10min_passed,
            "replay incorrectly certified live-camera throughput");

    RunMetrics short_run(26, start, 2, InputOrigin::LiveCamera);
    require(short_run.note(MetricStage::Capture, event(1, start - 1)) == EventDisposition::OutsideWindow,
            "warmup frame counted");
    require(short_run.note(MetricStage::Capture, event(1, start)) == EventDisposition::DuplicateOrReordered,
            "warmup duplicate counted at measurement boundary");
    require(short_run.note(MetricStage::Capture, event(2, start)) == EventDisposition::Counted, "start boundary");
    require(short_run.note(MetricStage::Capture, event(3, start + second)) == EventDisposition::Counted,
            "exact one-second boundary");
    require(short_run.note(MetricStage::Capture, event(3, start + second + 1)) ==
            EventDisposition::DuplicateOrReordered, "repeat counted");
    auto wrong_epoch = event(4, start + second + 10);
    wrong_epoch.key.epoch = 25;
    require(short_run.note(MetricStage::Capture, wrong_epoch) == EventDisposition::WrongEpoch, "old boot counted");
    auto bad_time = event(4, start + second + 10);
    bad_time.started_ns = bad_time.completed_ns + 1;
    require(short_run.note(MetricStage::Capture, bad_time) == EventDisposition::InvalidTime, "negative duration");
    require(short_run.note(MetricStage::Capture, event(4, start + 2 * second)) == EventDisposition::OutsideWindow,
            "exclusive final boundary counted");
    short_run.observe_pools(6, 2, 1);
    short_run.observe_pools(1, 1, 3);
    short_run.note_drop(DropReason::LatestReplaced, 17);
    const auto short_report = short_run.finalize(start + 2 * second);
    const auto& capture = short_report.stages[0];
    require(capture.windows == std::vector<std::uint32_t>({1, 1}) && capture.outside == 2 &&
            capture.duplicate_or_reordered == 2 && capture.wrong_epoch == 1 && capture.invalid_time == 1 &&
            capture.longest_gap_ns == second && capture.stage_duration->p95_ns == 10000000 &&
            capture.capture_to_completion->p99_ns == 20000000 &&
            short_report.pool_high_water == std::array<std::size_t, 3>({6, 2, 3}), "summary facts mismatch");

    RunMetrics early(26, start, 600, InputOrigin::LiveCamera);
    early.note(MetricStage::Inference, event(1, start + 100000000));
    const auto interrupted = early.finalize(start + second / 2);
    require(!interrupted.measurement_complete && !interrupted.camera_30fps_10min_passed &&
            interrupted.stages[inference_index].mean_fps == 2, "early stop claimed full run");

    RunMetrics bounded(26, start, 1, InputOrigin::Synthetic);
    for (unsigned i = 1; i <= 121; ++i) {
        bounded.note(MetricStage::Inference, event(i, start + i));
    }
    const auto bounded_report = bounded.finalize(start + second);
    require(bounded_report.stages[inference_index].new_frames == 121 &&
            bounded_report.stages[inference_index].latency_samples_omitted == 1 &&
            !bounded_report.stages[inference_index].capture_to_completion, "truncated latency claimed exact quantiles");

    RunMetrics concurrent(26, start, 1, InputOrigin::Synthetic);
    std::thread capture_thread([&] {
        for (unsigned i = 1; i <= 100; ++i) {
            concurrent.note(MetricStage::Capture, event(i, start + i));
        }
    });
    std::thread inference_thread([&] {
        for (unsigned i = 1; i <= 100; ++i) {
            concurrent.note(MetricStage::Inference, event(i, start + i));
        }
    });
    capture_thread.join();
    inference_thread.join();
    const auto concurrent_report = concurrent.finalize(start + second);
    require(concurrent_report.stages[0].new_frames == 100 &&
            concurrent_report.stages[inference_index].new_frames == 100, "concurrent stage updates lost");
    std::ostringstream output;
    output << std::hex;
    write_metrics_json(output, short_report);
    require(output.str().find("\"camera_30fps_10min_passed\":false") != std::string::npos &&
            output.str().find("\"windows\":[1,1]") != std::string::npos &&
            output.str().find("\"latest_replaced\":17") != std::string::npos, "report missing verification facts");
    std::cout << output.str();
    std::cout << "PASS metrics: 600-second traces, one-second windows, identity, early stop, bounded latency and concurrency\n";
}
