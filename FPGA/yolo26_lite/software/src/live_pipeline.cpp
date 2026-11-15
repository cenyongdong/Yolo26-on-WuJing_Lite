#include "yolo26/live_pipeline.hpp"
#include "yolo26/compose.hpp"
#include "yolo26/latest_slot.hpp"
#include "yolo26/video_engine.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <optional>
#include <stdexcept>
#include <thread>

namespace yolo26 {
namespace {

constexpr std::uint64_t kSecond = 1000000000;

std::string error_text(std::exception_ptr error) {
    if (!error) {
        return {};
    }
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& value) {
        return value.what();
    } catch (...) {
        return "non-standard worker exception";
    }
}

struct Workers {
    LatestSlot<FramePacket> capture, composed;
    LatestSlot<DetectionBundle> detections;
    std::atomic<bool> stop{false};
    std::thread camera_thread, composition_thread;
    CaptureCounters camera_counters{};
    bool camera_closed = false;
    std::exception_ptr camera_error, composition_error;

    void join() {
        stop.store(true);
        capture.close();
        detections.close();
        composed.close();
        if (camera_thread.joinable()) {
            camera_thread.join();
        }
        if (composition_thread.joinable()) {
            composition_thread.join();
        }
    }
    ~Workers() { join(); }
};

// 每个来源对象由其线程独占；这里只在完成后更新固定容量计数，不输出日志或访问硬件文件。
void capture_worker(std::unique_ptr<CameraCapture> camera, Workers& workers, FramePool& raw_pool,
                     RunMetrics& metrics, std::uint64_t deadline) {
    try {
        camera->start();
        while (!workers.stop.load() && monotonic_ns() < deadline) {
            auto frame = camera->next(raw_pool, 100);
            if (frame) {
                const auto& info = frame->info();
                metrics.note(MetricStage::Capture, {info.key, info.captured_ns, info.captured_ns, monotonic_ns()});
                workers.capture.publish(std::move(frame));
            }
        }
    } catch (const std::bad_alloc&) {
        workers.camera_error = std::current_exception();
        workers.stop.store(true);
    } catch (...) {
        // USB 断开只停止这个采集源；主线程仍能排空已完成帧及显示，并保存输入不足的实际计数。
        workers.camera_error = std::current_exception();
    }
    workers.camera_closed = camera->stop();
    workers.camera_counters = camera->counters();
}

void composition_worker(Workers& workers, FramePool& display_pool, RunMetrics& metrics) {
    try {
        Compositor720p compositor;
        while (!workers.stop.load()) {
            auto detection = workers.detections.wait_take();
            if (!detection) {
                break;
            }
            const auto begin = monotonic_ns();
            auto frame = compositor.compose(*detection, display_pool);
            if (!frame) {
                metrics.note_drop(DropReason::DisplayPoolFull);
                continue;
            }
            const auto& info = frame->info();
            metrics.note(MetricStage::Compose, {info.key, info.captured_ns, begin, monotonic_ns()});
            workers.composed.publish(std::move(frame));
        }
    } catch (...) {
        workers.composition_error = std::current_exception();
        workers.stop.store(true);
    }
}

bool drain_video(IcraftRuntime& runtime, VideoEngine& video, FramePool& model_pool) {
    if (!runtime.sdk_operations_complete()) {
        return false; // 不在未知 SDK DMA 后追加 MMIO，也不把 STOP 当作 SDK 传输完成。
    }
    video.request_stop();
    const auto deadline = monotonic_ns() + 2 * kSecond;
    while (video.must_retain_resources() && monotonic_ns() < deadline) {
        if (video.preprocess_busy()) {
            video.poll_preprocess(model_pool);
        }
        video.pump_display();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return !video.must_retain_resources(); // 期限仅结束观察，不释放内存或取消原硬件事务。
}

} // namespace

LiveReport run_live_pipeline(IcraftRuntime& runtime, const LiveConfig& config,
                             const std::function<bool()>& stop_requested) {
    if (config.seconds == 0 || config.seconds > 600 || config.warmup_seconds > 30 || !stop_requested ||
        !std::isfinite(config.threshold) || config.threshold < 0 || config.threshold > 1) {
        throw std::invalid_argument("invalid finite live pipeline configuration");
    }
    auto capture_config = config.capture;
    if (!capture_config.epoch) {
        capture_config.epoch = monotonic_ns();
    }
    auto camera = std::make_unique<CameraCapture>(capture_config);
    const auto format = camera->format();
    const auto color = resolve_capture_color(format, config.missing_color_fallback);
    if (format.size_image > 4 * 1024 * 1024 || (format.stride & 7U)) {
        throw std::invalid_argument("camera frame does not fit the aligned PL staging contract");
    }
    FramePool raw_pool(6, format.size_image), model_pool(3, runtime.model_geometry().rgb_elements());
    FramePool display_pool(3, Compositor720p::kBytes);
    LiveReport report{};
    report.format = format;
    report.color = color;
    const std::size_t record_limit = (config.seconds + config.warmup_seconds) * 120U;
    report.inference_times.reserve(record_limit);
    Workers workers; // 析构先关闭交接槽并 join；异常不能让 joinable thread 导致 terminate。
    auto& video = runtime.enable_video();
    // 设备初始化完成后才开始预热计时，不能让分配/寄存器检查占掉相机的测量时长。
    const auto start = monotonic_ns() + config.warmup_seconds * kSecond;
    const auto deadline = start + config.seconds * kSecond;
    RunMetrics metrics(capture_config.epoch, start, config.seconds, InputOrigin::LiveCamera);
    std::optional<PreparedFrame> prepared;
    std::optional<LiveFrameEvidence> pending_reference;
    std::uint64_t preprocess_started = 0;
    std::exception_ptr pipeline_error, drain_error;
    try {
        workers.camera_thread = std::thread(capture_worker, std::move(camera), std::ref(workers),
                                             std::ref(raw_pool), std::ref(metrics), deadline);
        workers.composition_thread = std::thread(composition_worker, std::ref(workers),
                                                  std::ref(display_pool), std::ref(metrics));
        while (!workers.stop.load() && monotonic_ns() < deadline) {
            if (stop_requested()) {
                report.interrupted = true;
                break;
            }
            if (const auto adopted = video.pump_display()) {
                const auto& timing = video.last_adoption();
                if (!timing || !(timing->frame.key == *adopted)) {
                    throw std::runtime_error("display adoption lost its original frame timing");
                }
                const auto observed_ns = monotonic_ns();
                metrics.note(MetricStage::Display, {timing->frame.key, timing->frame.captured_ns,
                             timing->upload_started_ns, observed_ns});
                if (pending_reference && pending_reference->source.key == *adopted) {
                    auto& reference = *pending_reference;
                    reference.adopted_ns = observed_ns;
                    reference.adoption = timing->snapshot;
                    reference.adopted_rgb565 = video.read_adopted_pixels(*adopted);
                    const bool same_pixels = reference.display_rgb565 == reference.adopted_rgb565;
                    report.reference = std::move(reference);
                    pending_reference.reset();
                    if (!same_pixels) {
                        throw std::runtime_error("adopted HDMI buffer differs from its same-frame composition");
                    }
                } else if (pending_reference && adopted->sequence > pending_reference->source.key.sequence) {
                    // 最新完整帧可能替换被选的样本；只把实际采用回执与同身份数据归档。
                    pending_reference.reset();
                }
            }
            if (video.preprocess_busy()) {
                auto result = video.poll_preprocess(model_pool);
                if (result) {
                    if (result->disposition == PreprocessDisposition::Ready) {
                        const auto& info = result->original->info();
                        metrics.note(MetricStage::Preprocess, {info.key, info.captured_ns,
                                     preprocess_started, monotonic_ns()});
                        if (prepared) {
                            metrics.note_drop(DropReason::LatestReplaced);
                        }
                        prepared = std::move(result);
                    } else if (result->disposition == PreprocessDisposition::NoHostBuffer) {
                        metrics.note_drop(DropReason::InputPoolFull);
                    }
                }
            }
            // 在进入阻塞的 NPU forward 前提交下一张 PL 预处理，让二者在不同硬件上重叠。
            if (!video.preprocess_busy()) {
                if (auto source = workers.capture.try_take()) {
                    preprocess_started = monotonic_ns();
                    video.begin_preprocess(std::move(source), color.matrix);
                }
            }
            if (auto frame = workers.composed.try_take()) {
                if (!video.upload_display(frame)) {
                    metrics.note_drop(DropReason::DisplayPoolFull);
                } else if (pending_reference && pending_reference->source.key == frame->info().key) {
                    pending_reference->display_rgb565.assign(frame->data(), frame->data() + frame->info().bytes_used);
                }
            }
            if (prepared) {
                const auto begin = monotonic_ns();
                const auto& rgb = prepared->rgb;
                auto result = runtime.infer_rgb(rgb->info().key, rgb->info().captured_ns,
                                                rgb->data(), rgb->info().bytes_used, rgb->info().stride,
                                                config.threshold);
                const bool sample = !pending_reference && (!report.reference ||
                    (report.reference->source_boxes.empty() && !result.detections.empty()));
                if (sample) {
                    const auto& source = prepared->original;
                    auto& reference = pending_reference.emplace();
                    reference.source = source->info();
                    reference.model_input = rgb->info();
                    reference.transform = prepared->geometry;
                    reference.inference_completed_ns = result.completed_ns;
                    reference.source_pixels.assign(source->data(), source->data() + source->info().bytes_used);
                    reference.model_rgb.assign(rgb->data(), rgb->data() + rgb->info().bytes_used);
                    reference.model_boxes = result.detections;
                    const auto raw = runtime.last_output();
                    if (!(raw.key == result.key) || !raw.bytes ||
                        raw.size != video.model_geometry().raw_elements() * sizeof(float)) {
                        throw std::runtime_error("raw detection evidence does not belong to its camera frame");
                    }
                    reference.raw_output.assign(raw.bytes, raw.bytes + raw.size);
                }
                auto detection = std::make_shared<DetectionBundle>(prepared->original, result.key,
                    prepared->geometry, prepared->matrix, std::move(result.detections), result.completed_ns);
                if (sample) {
                    pending_reference->source_boxes = detection->boxes();
                }
                // 有效新推理计数在数值/尾处理也成功后才提交，失败或非有限输出不产生有效结果。
                metrics.note(MetricStage::Inference, {result.key, result.captured_ns, begin, result.completed_ns});
                metrics.note(MetricStage::Postprocess, {result.key, result.captured_ns,
                             result.completed_ns - result.times.postprocess_ns, result.completed_ns});
                if (report.inference_times.size() < record_limit) {
                    report.inference_times.push_back({result.key, result.captured_ns, begin, result.completed_ns,
                                                      result.times});
                } else {
                    ++report.timing_records_omitted;
                }
                workers.detections.publish(std::move(detection));
                prepared.reset();
            }
            metrics.observe_pools(6 - raw_pool.available(), 3 - model_pool.available(), 3 - display_pool.available());
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    } catch (...) {
        pipeline_error = std::current_exception();
        // 保存当前输入及与它同身份的 SFB 失败快照；不可把前一帧输出归到这一次失败。
        if (prepared) {
            try {
                const auto& rgb = prepared->rgb;
                report.failed_input = rgb->info();
                report.failure_rgb.assign(rgb->data(), rgb->data() + rgb->info().bytes_used);
                const auto raw = runtime.last_output();
                if (raw.size && raw.key == rgb->info().key) {
                    report.failure_output.assign(raw.bytes, raw.bytes + raw.size);
                }
            } catch (...) {
                // 记录内存不足不能覆盖原失败，也不能改变硬件释放条件。
            }
        }
        metrics.note_drop(DropReason::DeviceError);
    }
    workers.join();
    try {
        report.video_drained = drain_video(runtime, video, model_pool);
    } catch (...) {
        drain_error = std::current_exception();
    }
    report.camera = workers.camera_counters;
    report.camera_closed = workers.camera_closed;
    report.camera_error = error_text(workers.camera_error);
    report.composition_error = error_text(workers.composition_error);
    report.pipeline_error = error_text(pipeline_error);
    report.drain_error = error_text(drain_error);
    if (!report.video_drained && report.drain_error.empty()) {
        report.drain_error = "video or SDK completion unproved; retain the runtime and inherited board lease";
    }
    metrics.note_drop(DropReason::LatestReplaced, workers.capture.dropped() + workers.detections.dropped() +
                     workers.composed.dropped() + workers.camera_counters.latest_replaced +
                     video.replaced_display_frames());
    metrics.note_drop(DropReason::CapturePoolFull, workers.camera_counters.pool_exhausted);
    metrics.observe_pools(raw_pool.high_water(), model_pool.high_water(), display_pool.high_water());
    // 预热中收到停止或错误时，测量区间尚未开始，按零时长结算而不是抛出第二个时间错误。
    report.metrics = metrics.finalize(std::max(start, monotonic_ns()));
    return report;
}

} // namespace yolo26
