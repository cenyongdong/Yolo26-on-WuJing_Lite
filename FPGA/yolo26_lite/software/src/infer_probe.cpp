#include "yolo26/icraft_runtime.hpp"
#include "yolo26/live_pipeline.hpp"
#include "yolo26/metrics.hpp"
#include "yolo26/preprocess.hpp"
#include <icraft-xir/utils/json.hpp>
#include <charconv>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
volatile std::sig_atomic_t stop_requested = 0;

void stop_handler(int) {
    stop_requested = 1;
}

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

unsigned number(const char* text, unsigned minimum, unsigned maximum) {
    const std::string value(text);
    unsigned result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
            result >= minimum && result <= maximum, "invalid finite probe argument");
    return result;
}

Json read_json(const fs::path& path) {
    std::ifstream input(path);
    require(bool(input), "required board launch record is absent");
    Json value;
    input >> value;
    return value;
}

// O_EXCL 保留原始失败/输出；JSON 序列化由 SDK 附带的公开 nlohmann 头处理。
void save_bytes(const fs::path& path, const char* bytes, std::size_t size) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot create a new native evidence record");
    std::size_t written = 0;
    while (written < size) {
        const auto count = ::write(fd, bytes + written, size - written);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error("native evidence write failed");
        }
        written += static_cast<std::size_t>(count);
    }
    require(::close(fd) == 0, "native evidence close failed");
}

void save_new(const fs::path& path, const Json& value) {
    const auto bytes = value.dump(2) + "\n";
    save_bytes(path, bytes.data(), bytes.size());
}

std::string boot_id() {
    std::ifstream input("/proc/sys/kernel/random/boot_id");
    std::string value;
    std::getline(input, value);
    require(bool(input) && value.size() == 36, "current BOOT identity is unavailable");
    return value;
}

std::string own_start_ticks() {
    std::ifstream input("/proc/self/stat");
    std::string line;
    std::getline(input, line);
    const auto end_name = line.rfind(") ");
    require(end_name != std::string::npos, "process start identity is unavailable");
    std::istringstream fields(line.substr(end_name + 2));
    std::string value;
    for (unsigned index = 0; index <= 19; ++index) {
        require(bool(fields >> value), "process start identity is incomplete");
    }
    return value;
}

struct LaunchOptions {
    yolo26::IcraftInputProfile input_profile;
    int ocm_strategy;
    unsigned diagnostic_samples;
    yolo26::IcraftModelVariant model_variant;
    std::size_t candidate_hardops;
    yolo26::IcraftOutputRead output_read;
    bool counter_trace;
    bool bus_trace;
    yolo26::ModelGeometry geometry;
    unsigned replay_frames;
    std::string camera_bus_info;
    float score_threshold;
};

LaunchOptions verify_launch(int lease_fd, const std::string& expected_boot, const fs::path& directory,
                            unsigned budget, const char* source, const char* source_path) {
    const bool live_camera = std::string(source) == "live_camera";
    const bool sequence = std::string(source) == "file_sequence";
    struct stat owned{}, shared{};
    require(::fstat(lease_fd, &owned) == 0 && ::stat("/run/pose180-resident.lock", &shared) == 0 &&
            S_ISREG(owned.st_mode) && owned.st_dev == shared.st_dev && owned.st_ino == shared.st_ino &&
            ::flock(lease_fd, LOCK_EX | LOCK_NB) == 0, "shared board lock was not inherited");
    require(boot_id() == expected_boot, "BOOT changed before native startup");
    // 父进程先持锁预检，再启动子进程并保存 PID/start_ticks，最后以原子重命名发布本次唯一许可。
    // 等待期限只发生在任何 SDK 设备调用之前；不以超时取消已提交的硬件负载。
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto permit_path = directory / "startup-permit.json";
    while (!fs::exists(permit_path)) {
        require(!stop_requested && std::chrono::steady_clock::now() < deadline, "native startup permit not published");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto active = read_json("/run/pose180-resident.active.json");
    const auto process = read_json(directory / "worker-process.json");
    const auto permit = read_json(permit_path);
    const auto input_profile = permit.value("input_profile", std::string("fp32"));
    require(input_profile == "fp32" || input_profile == "rgb_u8", "unknown permitted model input profile");
    const auto strategy = permit.value("ocm_strategy", -1);
    const auto output_read = permit.value("output_read", std::string("sfb"));
    const auto variant = permit.value("model_variant", std::string("baseline"));
    const bool candidate = variant != "baseline";
    const yolo26::ModelGeometry geometry{permit.value("input_width", 640U), permit.value("input_height", 640U)};
    const bool rectangular = variant == "int8_psa_rect" && geometry == yolo26::kRectangularModel;
    const bool rectangular_live = live_camera && rectangular && permit.value("qualified_rectangular_video", false);
    require(yolo26::is_square_model(geometry) || (rectangular && (!live_camera || rectangular_live)),
            "unexpected permitted input dimensions");
    require(output_read == "sfb" ||
            (output_read == "plain_fp32" && strategy == -1 && (!live_camera || rectangular_live)),
            "direct output mode differs from the qualified I/O profile");
    // Python 包契约先核验登记名及完整 JSON/RAW 哈希；许可只传递已登记的图规模。
    // 仍在任何设备调用前验证实际图节点数、公开端口及 Host 分配，不能仅凭许可跳过审查。
    require(!candidate || (permit.value("registered_candidate", false) &&
            permit.contains("candidate_hardops") && permit.at("candidate_hardops").is_number_unsigned()),
            "candidate is absent from the verified package registry");
    const auto candidate_hardops = candidate ? permit.at("candidate_hardops").get<std::size_t>() : 0;
    require(!candidate || (candidate_hardops > 0 && candidate_hardops <= 4096), "invalid registered graph count");
    require(!candidate || (input_profile == "fp32" && strategy == -1 && (!live_camera || rectangular_live)),
            "candidate model, source or OCM permission differs");
    const auto samples = permit.value("diagnostic_samples", 0);
    const bool counter_trace = permit.value("counter_trace", false);
    const bool bus_trace = permit.value("bus_trace", false);
    require(!counter_trace || (!live_camera && samples > 0), "counter tracing needs file diagnostic samples");
    require(!bus_trace || (!live_camera && samples > 0), "bus tracing needs file diagnostic samples");
    require(strategy >= -1 && strategy <= 3 && samples >= 0 &&
            static_cast<unsigned>(samples) <= std::min(budget, 8U) &&
            (!live_camera || (strategy == -1 && samples == 0)), "invalid permitted OCM diagnostic scope");
    require(!sequence || (variant == "int8_psa_rect" && output_read == "plain_fp32" && samples == 0),
            "timed sequence requires the rectangular graph and direct FP32 read");
    const auto replay_frames = permit.value("replay_frames", 1U);
    require(sequence ? replay_frames >= 2 && replay_frames <= 32 : replay_frames == 1,
            "invalid RGB sequence length");
    const auto ticks = process.at("start_ticks").is_string() ? process.at("start_ticks").get<std::string>() :
                       std::to_string(process.at("start_ticks").get<std::uint64_t>());
    require(active.at("boot_id") == expected_boot && fs::canonical(active.at("run").get<std::string>()) == directory &&
            process.at("pid").get<int>() == ::getpid() &&
            ticks == own_start_ticks() &&
            process.at("boot_id") == expected_boot && permit.at("boot_id") == expected_boot &&
            permit.at("sdk_starts").get<unsigned>() == 1 &&
            permit.at("model_identity_verified").get<bool>() && permit.at("previous_startup_unused").get<bool>(),
            "native process, startup permit or protected model identity differs");
    require(permit.value("source", std::string("file_replay")) == source &&
            permit.value("capture_starts", 0) == (live_camera ? 1 : 0), "native source permission differs");
    if (live_camera || sequence) {
        require(permit.at("duration_seconds").get<unsigned>() == budget, "timed native budget differs");
    } else {
        require(permit.at("iterations").get<unsigned>() == budget, "file replay iteration budget differs");
    }
    std::string camera_bus;
    float score_threshold = 0.25F;
    if (live_camera) {
        require(permit.at("warmup_seconds") == 2, "live camera warmup differs");
        const auto permitted_device = permit.value("video_device", std::string("/dev/video1"));
        constexpr std::string_view prefix = "/dev/video";
        const std::string_view node = source_path;
        require(node == permitted_device && node.size() > prefix.size() && node.size() <= prefix.size() + 4 &&
                node.substr(0, prefix.size()) == prefix &&
                std::all_of(node.begin() + prefix.size(), node.end(), [](char c) { return c >= '0' && c <= '9'; }),
                "native video device differs from preflight");
        camera_bus = permit.value("video_bus_info", std::string());
        require(!rectangular_live || (!camera_bus.empty() && camera_bus.size() < 32), "missing USB port identity");
        if (rectangular_live) {
            score_threshold = permit.value("score_threshold", 0.25F);
            require(std::isfinite(score_threshold) && score_threshold >= 0.0F && score_threshold <= 1.0F,
                    "invalid permitted live score threshold");
        }
    }
    require(!rectangular_live || output_read == "plain_fp32", "rectangular video requires direct FP32 readback");
    save_new(directory / "sdk-start-claimed.json", {{"boot_id", expected_boot}, {"pid", ::getpid()},
             {"start_ticks", own_start_ticks()}, {"sdk_start_budget", 1}, {"automatic_retry", false},
             {"source", source}, {"input_profile", input_profile},
             {"ocm_strategy", strategy}, {"diagnostic_samples", samples}, {"model_variant", variant},
             {"candidate_hardops", candidate_hardops}, {"output_read", output_read},
             {"counter_trace", counter_trace}, {"bus_trace", bus_trace},
             {"input_width", geometry.width}, {"input_height", geometry.height}});
    const auto selected_variant = candidate ? yolo26::IcraftModelVariant::RegisteredCandidate :
                                             yolo26::IcraftModelVariant::Baseline;
    return {input_profile == "rgb_u8" ? yolo26::IcraftInputProfile::RgbUInt8 :
                                       yolo26::IcraftInputProfile::Fp32Normalized,
            strategy, static_cast<unsigned>(samples), selected_variant, candidate_hardops,
            output_read == "plain_fp32" ? yolo26::IcraftOutputRead::PlainFp32 : yolo26::IcraftOutputRead::Sfb,
            counter_trace, bus_trace, geometry, replay_frames, std::move(camera_bus), score_threshold};
}

void save_raw(const fs::path& path, const yolo26::RawOutputView& view) {
    save_bytes(path, view.bytes, view.size);
}

Json times_json(const yolo26::InferenceTimes& times) {
    return {{"normalize_ns", times.normalize_ns}, {"input_write_ns", times.input_write_ns},
            {"clear_ns", times.clear_ns}, {"forward_ns", times.forward_ns}, {"final_wait_ns", times.final_wait_ns},
            {"output_read_ns", times.output_read_ns}, {"postprocess_ns", times.postprocess_ns}};
}

Json boxes_json(const std::vector<yolo26::Detection>& detections) {
    Json boxes = Json::array();
    for (const auto& box : detections) {
        boxes.push_back({box.x1, box.y1, box.x2, box.y2, box.score, box.class_id});
    }
    return boxes;
}

struct SlowReplayCall {
    std::uint64_t frame = 0, start_ns = 0, complete_ns = 0;
    std::uint64_t thread_cpu_ns = 0;
    long major_faults = 0, voluntary_switches = 0, involuntary_switches = 0;
    unsigned input_slot = 0;
    yolo26::InferenceTimes times;

    std::uint64_t elapsed_ns() const { return complete_ns - start_ns; }
};

std::uint64_t cpu_ns(const rusage& usage) {
    const auto seconds = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec;
    const auto micros = usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
    return std::uint64_t(seconds) * 1000000000 + std::uint64_t(micros) * 1000;
}

void run_sequence(yolo26::IcraftRuntime& runtime, const fs::path& directory, unsigned seconds,
                  yolo26::ModelGeometry geometry, const std::vector<std::uint8_t>& rgb, unsigned frames) {
    using namespace yolo26;
    constexpr std::uint64_t second_ns = 1000000000;
    const auto frame_bytes = geometry.rgb_elements();
    const auto raw_bytes = geometry.raw_elements() * sizeof(float);
    const auto epoch = now_ns();
    std::uint64_t sequence = 0;
    std::vector<char> references(raw_bytes * frames);
    auto infer = [&](unsigned slot) {
        const FrameKey key{epoch, sequence++};
        const auto result = runtime.infer_rgb(key, now_ns(), rgb.data() + frame_bytes * slot, frame_bytes,
                                              geometry.rgb_row_bytes());
        const auto raw = runtime.last_output();
        require(raw.bytes && raw.size == raw_bytes && raw.key == key && result.key == key &&
                result.host_compute_callbacks == 0, "file sequence result has the wrong identity or layout");
        return result;
    };
    auto compare = [&](unsigned slot) {
        const auto raw = runtime.last_output();
        const auto* reference = references.data() + slot * raw_bytes;
        if (std::memcmp(reference, raw.bytes, raw_bytes) != 0) {
            save_raw(directory / "native-sequence-different-output.f32", raw);
            save_bytes(directory / "native-sequence-expected-output.f32", reference, raw_bytes);
            save_new(directory / "native-sequence-different-output.json",
                     {{"input_slot", slot}, {"frame", raw.key.sequence}});
            throw std::runtime_error("file sequence full output differs from the same-input warmup");
        }
    };

    // 预热及计时均保留参考到内存，避免预热张量的异步文件回写干扰计时。
    unsigned warmed = 0;
    for (; warmed < frames && !stop_requested; ++warmed) {
        infer(warmed);
        std::memcpy(references.data() + warmed * raw_bytes, runtime.last_output().bytes, raw_bytes);
    }

    const auto start = now_ns();
    RunMetrics metrics(epoch, start, seconds, InputOrigin::FileReplay);
    const auto deadline = start + seconds * second_ns;
    InferenceTimes totals{};
    std::uint64_t timed = 0, checked = 0, slow_count = 0;
    std::array<SlowReplayCall, 16> slow_calls{};
    unsigned slot = 0, check_slot = 0;
    while (!stop_requested && now_ns() < deadline) {
        rusage before{};
        require(::getrusage(RUSAGE_THREAD, &before) == 0, "cannot sample replay thread CPU time");
        const auto result = infer(slot);
        if (result.completed_ns - result.captured_ns >= 40000000) {
            rusage after{};
            require(::getrusage(RUSAGE_THREAD, &after) == 0, "cannot sample slow replay call");
            ++slow_count;
            const auto shortest = std::min_element(slow_calls.begin(), slow_calls.end(),
                [](const SlowReplayCall& a, const SlowReplayCall& b) { return a.elapsed_ns() < b.elapsed_ns(); });
            if (result.completed_ns - result.captured_ns > shortest->elapsed_ns()) {
                *shortest = {result.key.sequence, result.captured_ns, result.completed_ns,
                             cpu_ns(after) - cpu_ns(before), after.ru_majflt - before.ru_majflt,
                             after.ru_nvcsw - before.ru_nvcsw, after.ru_nivcsw - before.ru_nivcsw,
                             slot, result.times};
            }
        }
        // 每轮轮换检查一个来源。每帧均由解码器扫描完整张量的有限值。
        if (slot == check_slot) {
            compare(slot);
            ++checked;
        }
        const auto event = metrics.note(MetricStage::Inference,
            {result.key, result.captured_ns, result.captured_ns, result.completed_ns});
        require(event == EventDisposition::Counted || event == EventDisposition::OutsideWindow,
                "invalid repeated file sequence event");
        if (event == EventDisposition::Counted) {
            totals.normalize_ns += result.times.normalize_ns;
            totals.input_write_ns += result.times.input_write_ns;
            totals.clear_ns += result.times.clear_ns;
            totals.forward_ns += result.times.forward_ns;
            totals.final_wait_ns += result.times.final_wait_ns;
            totals.output_read_ns += result.times.output_read_ns;
            totals.postprocess_ns += result.times.postprocess_ns;
            ++timed;
        }
        if (++slot == frames) {
            slot = 0;
            check_slot = (check_slot + 1) % frames;
        }
    }
    const auto report = metrics.finalize(now_ns());
    unsigned verified = 0;
    for (; verified < warmed && !stop_requested; ++verified) {
        infer(verified);
        compare(verified);
    }
    require(runtime.close(), "file sequence SDK shutdown is unproved");
    for (unsigned index = 0; index < warmed; ++index) {
        save_bytes(directory / ("native-sequence-input-" + std::to_string(index) + ".f32"),
                   references.data() + index * raw_bytes, raw_bytes);
    }
    std::sort(slow_calls.begin(), slow_calls.end(), [](const SlowReplayCall& a, const SlowReplayCall& b) {
        return a.elapsed_ns() > b.elapsed_ns();
    });
    Json slow = Json::array();
    for (const auto& call : slow_calls) {
        if (call.elapsed_ns() == 0) {
            break;
        }
        slow.push_back({{"frame", call.frame}, {"input_slot", call.input_slot}, {"start_ns", call.start_ns},
            {"complete_ns", call.complete_ns}, {"wall_ns", call.elapsed_ns()}, {"thread_cpu_ns", call.thread_cpu_ns},
            {"major_faults", call.major_faults}, {"voluntary_switches", call.voluntary_switches},
            {"involuntary_switches", call.involuntary_switches}, {"stages", times_json(call.times)}});
    }
    std::ostringstream encoded;
    write_metrics_json(encoded, report);
    save_new(directory / "native-sequence-summary.json", {{"source", "file_sequence"},
        {"distinct_rgb_frames", frames}, {"warmup_frames", warmed}, {"checked_after_measurement", verified},
        {"checks_during_measurement", checked}, {"stable_full_outputs", true}, {"interrupted", stop_requested != 0},
        {"timed_calls", timed}, {"stage_totals_ns", times_json(totals)}, {"slow_calls_at_least_40ms", slow_count},
        {"longest_calls", slow}, {"metrics", Json::parse(encoded.str())}});
    save_new(directory / "native-complete.json", {{"boot_id", boot_id()}, {"source", "file_sequence"},
        {"completed", timed}, {"interrupted", stop_requested != 0}, {"native_shutdown_complete", true},
        {"allocation_count_after", 0}, {"sdk_starts", 1}, {"capture_starts", 0}, {"boot_writes", 0}, {"reboots", 0},
        {"camera_30fps_claim", false}});
}

void run_live(yolo26::IcraftRuntime& runtime, const fs::path& directory, const char* device,
              const LaunchOptions& launch, unsigned seconds) {
    yolo26::LiveConfig config;
    config.capture.device = device;
    config.capture.expected_bus_info = launch.camera_bus_info;
    config.seconds = seconds;
    config.threshold = launch.score_threshold;
    const auto report = yolo26::run_live_pipeline(runtime, config, [] { return stop_requested != 0; });
    std::ostringstream metrics;
    yolo26::write_metrics_json(metrics, report.metrics);
    const auto& format = report.format;
    const auto& camera = report.camera;
    const auto display_index = static_cast<std::size_t>(yolo26::MetricStage::Display);
    const auto capture_index = static_cast<std::size_t>(yolo26::MetricStage::Capture);
    const auto preproc_index = static_cast<std::size_t>(yolo26::MetricStage::Preprocess);
    const auto every_second = [&](std::size_t index) {
        const auto& windows = report.metrics.stages[index].windows;
        return std::all_of(windows.begin(), windows.end(), [](auto count) { return count >= 30; });
    };
    const auto full_chain_pass = report.metrics.camera_30fps_10min_passed && every_second(capture_index) &&
                                 every_second(preproc_index) && every_second(display_index);
    const auto same_frame = report.reference && report.reference->display_rgb565 == report.reference->adopted_rgb565;
    const Json summary{{"source", "live_camera"}, {"video_device", device},
        {"video_bus_info", launch.camera_bus_info}, {"score_threshold", config.threshold},
        {"metrics", Json::parse(metrics.str())},
        {"camera_closed", report.camera_closed}, {"video_drained", report.video_drained},
        {"complete_chain_30fps_10min_passed", full_chain_pass},
        {"adopted_reference_matches", same_frame},
        {"adopted_reference_boxes", report.reference ? report.reference->source_boxes.size() : 0},
        {"interrupted", report.interrupted}, {"camera_error", report.camera_error},
        {"composition_error", report.composition_error}, {"pipeline_error", report.pipeline_error},
        {"drain_error", report.drain_error}, {"hdmi_refresh_counter_available", false},
        {"timing_records_omitted", report.timing_records_omitted},
        {"format", {{"width", format.width}, {"height", format.height}, {"stride", format.stride},
            {"size_image", format.size_image}, {"buffers", format.buffers}, {"colorspace", format.colorspace},
            {"ycbcr_encoding", format.ycbcr_encoding}, {"quantization", format.quantization},
            {"transfer_function", format.transfer_function}, {"extended_fields_valid", format.extended_fields_valid}}},
        {"conversion", {{"matrix", static_cast<unsigned>(report.color.matrix)},
            {"assumed_missing_metadata", report.color.assumed_missing_metadata}}},
        {"capture", {{"dequeued", camera.dequeued}, {"published", camera.published},
            {"driver_sequence_gaps", camera.driver_sequence_gaps}, {"invalid_bytes", camera.invalid_bytes},
            {"frame_interval_numerator", format.frame_interval_numerator},
            {"frame_interval_denominator", format.frame_interval_denominator},
            {"invalid_timestamp", camera.invalid_timestamp}, {"duplicate_or_reordered", camera.duplicate_or_reordered},
            {"latest_replaced", camera.latest_replaced}, {"pool_exhausted", camera.pool_exhausted}}}};
    // 视频循环已停止，先保存失败也需要的有界摘要；没有把磁盘写入放入采集或 forward 路径。
    save_new(directory / "native-live-summary.json", summary);
    if (report.failed_input) {
        const auto& frame = *report.failed_input;
        save_new(directory / "native-live-failure-frame.json", {{"epoch", frame.key.epoch},
                 {"frame", frame.key.sequence}, {"captured_ns", frame.captured_ns},
                 {"output_is_failure_snapshot", true}, {"output_bytes", report.failure_output.size()}});
        save_bytes(directory / "native-live-failure-rgb.bin", reinterpret_cast<const char*>(report.failure_rgb.data()),
                   report.failure_rgb.size());
        save_bytes(directory / "native-live-failure-output.bin", report.failure_output.data(), report.failure_output.size());
    }
    require(report.video_drained && runtime.close(), "live video or SDK shutdown is unproved; retain owner");
    if (report.reference) {
        const auto& reference = *report.reference;
        const auto& frame = reference.source;
        const auto& transform = reference.transform;
        save_new(directory / "native-reference-frame.json", {{"epoch", frame.key.epoch}, {"frame", frame.key.sequence},
                 {"captured_ns", frame.captured_ns}, {"width", frame.width}, {"height", frame.height},
                 {"stride", frame.stride}, {"bytes_used", frame.bytes_used},
                 {"model_width", reference.model_input.width}, {"model_height", reference.model_input.height},
                 {"model_stride", reference.model_input.stride}, {"model_bytes", reference.model_input.bytes_used},
                 {"resized_width", transform.resized_width}, {"resized_height", transform.resized_height},
                 {"left", transform.left}, {"top", transform.top}, {"gain", transform.gain},
                 {"inference_completed_ns", reference.inference_completed_ns}, {"adopted_ns", reference.adopted_ns},
                 {"adopted_hardware_tag", reference.adoption.accepted_id},
                 {"adopted_address", reference.adoption.accepted_address},
                 {"adopted_status", reference.adoption.status}, {"adopted_faults", reference.adoption.faults},
                 {"model_boxes", boxes_json(reference.model_boxes)}, {"source_boxes", boxes_json(reference.source_boxes)}});
        auto archive_pixels = [&](const char* name, const std::vector<std::uint8_t>& data) {
            save_bytes(directory / name, reinterpret_cast<const char*>(data.data()), data.size());
        };
        archive_pixels("native-reference-source.yuyv", reference.source_pixels);
        archive_pixels("native-reference-rgb.bin", reference.model_rgb);
        archive_pixels("native-reference-composed.rgb565", reference.display_rgb565);
        archive_pixels("native-reference-scan-buffer.rgb565", reference.adopted_rgb565);
        save_bytes(directory / "native-reference-output.f32", reference.raw_output.data(), reference.raw_output.size());
    }
    Json times = Json::array();
    for (const auto& frame : report.inference_times) {
        times.push_back({{"epoch", frame.key.epoch}, {"frame", frame.key.sequence}, {"captured_ns", frame.captured_ns},
                         {"started_ns", frame.started_ns}, {"completed_ns", frame.completed_ns},
                         {"times", times_json(frame.stages)}});
    }
    save_new(directory / "native-live-times.json", times);
    require(report.camera_closed && report.camera_error.empty() && report.composition_error.empty() &&
            report.pipeline_error.empty(), "live run had a source, composition or inference failure");
    require(report.interrupted || (!report.inference_times.empty() && report.metrics.stages[display_index].new_frames > 0 &&
            same_frame), "live run completed without a checked same-frame HDMI adoption");
    const auto index = static_cast<std::size_t>(yolo26::MetricStage::Inference);
    save_new(directory / "native-complete.json", {{"boot_id", boot_id()}, {"source", "live_camera"},
             {"completed", report.metrics.stages[index].new_frames}, {"interrupted", report.interrupted},
             {"native_shutdown_complete", true}, {"allocation_count_after", 0}, {"sdk_starts", 1},
             {"capture_starts", 1}, {"boot_writes", 0}, {"reboots", 0}, {"duration_seconds", seconds},
             {"camera_30fps_claim", report.metrics.camera_30fps_10min_passed}});
}

} // namespace

int main(int argc, char** argv) {
    const bool live_camera = argc == 9 && std::string(argv[8]) == "--live-camera";
    const bool sequence = argc == 9 && std::string(argv[8]) == "--replay-sequence";
    if (argc != 8 && !live_camera && !sequence) {
        std::cerr << "usage: yolo26_infer_probe LEASE_FD BOOT_ID GRAPH RAW INPUT_RGB RUN_DIR ITERATIONS(1..30)\n"
                     "Requires the owned-board launcher and its single-use startup permit.\n";
        std::cerr << "live mode: replace INPUT_RGB with the verified V4L2 capture node and ITERATIONS with SECONDS(1..600),"
                     " then append --live-camera. A matching live-camera permit is mandatory.\n";
        std::cerr << "sequence mode: pass the packed RGB file and bounded seconds, then --replay-sequence.\n";
        return 2;
    }
    std::unique_ptr<yolo26::IcraftRuntime> runtime;
    fs::path directory;
    try {
        const int lease_fd = static_cast<int>(number(argv[1], 3, 4096));
        const unsigned iterations = number(argv[7], 1, live_camera || sequence ? 600 : 30);
        directory = fs::canonical(argv[6]);
        require(directory.string().find("/root/yolo26-") == 0 && fs::is_directory(directory),
                "probe run directory is outside its isolated board workspace");
        struct sigaction action{};
        action.sa_handler = stop_handler;
        ::sigemptyset(&action.sa_mask);
        for (const int signal : {SIGINT, SIGTERM, SIGHUP}) {
            require(::sigaction(signal, &action, nullptr) == 0, "cannot install non-destructive stop handling");
        }
        const char* source = live_camera ? "live_camera" : sequence ? "file_sequence" : "file_replay";
        const auto launch = verify_launch(lease_fd, argv[2], directory, iterations, source, argv[5]);
        const auto bytes = launch.geometry.rgb_elements() * launch.replay_frames;
        std::vector<std::uint8_t> rgb;
        if (!live_camera) {
            std::ifstream input(argv[5], std::ios::binary | std::ios::ate);
            require(input && input.tellg() == std::streampos(bytes), "RGB fixture differs from the permitted geometry");
            rgb.resize(bytes);
            input.seekg(0);
            input.read(reinterpret_cast<char*>(rgb.data()), rgb.size());
            require(bool(input), "RGB fixture read failed");
        }
        require(!stop_requested, "stop requested before SDK initialization");
        runtime = std::make_unique<yolo26::IcraftRuntime>();
        yolo26::IcraftConfig config{argv[3], argv[4], [directory](const char* stage) {
            save_new(directory / "native-retained.json", {{"stage", stage}, {"pid", ::getpid()},
                     {"boot_id", boot_id()}, {"resources_retained", true}, {"automatic_retry", false}});
        }};
        config.input_profile = launch.input_profile;
        config.ocm_strategy = launch.ocm_strategy;
        config.model_variant = launch.model_variant;
        config.candidate_hardops = launch.candidate_hardops;
        config.output_read = launch.output_read;
        config.diagnostics = launch.diagnostic_samples != 0 || launch.ocm_strategy != -1;
        config.bus_trace = launch.bus_trace;
        config.geometry = launch.geometry;
        const auto identity = runtime->initialize(config);
        Json fusion = Json::array();
        for (const auto& group : identity.fusion) {
            fusion.push_back({{"operation", group.operation}, {"first_layer", group.first_layer},
                              {"layer_count", group.layer_count}, {"originals", group.originals}});
        }
        save_new(directory / "native-initialized.json", {{"device", identity.device}, {"icore", identity.icore},
                 {"completion_layers", identity.completion_layers}, {"fusion", fusion}, {"sdk_starts", 1},
                 {"input_profile", identity.input_profile}, {"logical_input_bytes", identity.logical_input_bytes},
                 {"input_transfer", identity.input_transfer},
                 {"normalization_backend", yolo26::normalization_backend()},
                 {"input_width", launch.geometry.width}, {"input_height", launch.geometry.height},
                 {"output_anchors", launch.geometry.anchors()}});
        if (config.diagnostics) {
            save_new(directory / "npu-deployment.json", Json::parse(runtime->deployment_diagnostics()));
        }
        if (live_camera) {
            run_live(*runtime, directory, argv[5], launch, iterations);
            return 0;
        }
        if (sequence) {
            run_sequence(*runtime, directory, iterations, launch.geometry, rgb, launch.replay_frames);
            return 0;
        }
        const std::uint64_t epoch = now_ns();
        unsigned completed = 0;
        std::vector<char> first_output;
        for (unsigned index = 0; index < iterations && !stop_requested; ++index) {
            const bool diagnostic = index >= iterations - launch.diagnostic_samples;
            if (diagnostic && launch.counter_trace) {
                runtime->arm_counter_trace();
            }
            if (diagnostic && launch.bus_trace) {
                runtime->arm_bus_trace();
            }
            const auto result = runtime->infer_rgb({epoch, index}, now_ns(), rgb.data(), rgb.size(),
                                                   launch.geometry.rgb_row_bytes());
            const auto raw = runtime->last_output();
            if (index == 0) {
                save_raw(directory / "native-first-output.f32", raw);
                first_output.assign(raw.bytes, raw.bytes + raw.size);
            }
            const bool same_output = raw.size == first_output.size() &&
                                     std::equal(first_output.begin(), first_output.end(), raw.bytes);
            if (!same_output) {
                save_raw(directory / ("native-different-output-" + std::to_string(index) + ".f32"), raw);
            }
            if (diagnostic) {
                save_new(directory / ("npu-timers-" + std::to_string(index) + ".json"),
                         Json::parse(runtime->collect_npu_diagnostics()));
            }
            const Json record{{"source", "file_replay"}, {"epoch", result.key.epoch}, {"frame", result.key.sequence},
                              {"captured_ns", result.captured_ns}, {"completed_ns", result.completed_ns},
                              {"completed_layers", result.completed_layers}, {"npu_callbacks", result.npu_callbacks},
                              {"host_compute_callbacks", result.host_compute_callbacks}, {"times", times_json(result.times)},
                              {"detections", boxes_json(result.detections)}, {"new_camera_frames", 0}, {"npu_executed", true},
                              {"same_as_first_raw", same_output}, {"timer_sample_collected", diagnostic},
                              {"ocm_strategy", launch.ocm_strategy},
                              {"counter_trace_collected", diagnostic && launch.counter_trace},
                              {"bus_trace_collected", diagnostic && launch.bus_trace}};
            save_new(directory / ("native-frame-" + std::to_string(index) + ".json"), record);
            require(same_output, "fixed-input raw output changed within one OCM strategy; preserved full output");
            ++completed;
            std::cout << record.dump() << '\n' << std::flush;
        }
        require(runtime->close(), "SDK shutdown is unproved; retain process and inherited lease");
        save_new(directory / "native-complete.json", {{"boot_id", boot_id()}, {"completed", completed},
                 {"interrupted", stop_requested != 0}, {"native_shutdown_complete", true},
                 {"allocation_count_after", 0}, {"sdk_starts", 1}, {"capture_starts", 0}, {"boot_writes", 0},
                 {"reboots", 0}, {"camera_30fps_claim", false}});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native YOLO probe stopped: " << error.what() << '\n';
        if (runtime && !directory.empty()) {
            try {
                save_raw(directory / "native-failure-output.bin", runtime->last_output());
                save_new(directory / "native-failure.json", {{"error", error.what()},
                         {"release_proven", runtime->release_proven()}, {"pending_stage", runtime->pending_stage()},
                         {"automatic_retry", false}, {"boot_id", boot_id()}});
            } catch (...) {
                // 保存失败不覆盖原异常，也不放宽未知硬件事务的释放条件。
            }
            try {
                if (runtime->close()) {
                    save_new(directory / "native-failed-closed.json", {{"boot_id", boot_id()},
                             {"native_shutdown_complete", true}, {"allocation_count_after", 0},
                             {"original_failure_preserved", true}, {"automatic_retry", false}});
                }
            } catch (...) {
                // close 内部保留未知释放状态；下方析构不能再尝试释放该对象。
            }
        }
        // 如果 SDK 操作状态未知，析构中的既有 ReleaseGuard 会保留整个 owner，而不会退出进程。
        runtime.reset();
        return 1;
    }
}
