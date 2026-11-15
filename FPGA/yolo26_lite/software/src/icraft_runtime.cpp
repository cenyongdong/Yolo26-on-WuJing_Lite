#include "yolo26/icraft_runtime.hpp"
#include "yolo26/preprocess.hpp"
#include "yolo26/npu_trace.hpp"
#include "release_guard.hpp"
#include "icraft_video_io.hpp"
#include "icraft_diagnostics.hpp"
#include "icraft_tensor_io.hpp"
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xir/ops/ops.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <thread>

namespace yolo26 {
namespace {

namespace xir = icraft::xir;
using namespace icraft::xrt;
using Clock = std::chrono::steady_clock;
std::atomic_flag startup_consumed = ATOMIC_FLAG_INIT;

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

bool shape_is(const xir::TensorType& type, std::initializer_list<std::int64_t> expected) {
    return type->shape.size() == expected.size() &&
           std::equal(type->shape.begin(), type->shape.end(), expected.begin());
}

// 对已存在的 float 对象按字符表示写入，避免把 byte vector 猜测为合法 float 对象。
// 额外一个字节用于检测超长 dump；不让 SDK 扩容、不解析硬件私有布局、不重复读取输出。
class TensorSink : public std::streambuf {
public:
    explicit TensorSink(std::vector<float>& storage) : storage_(storage) {}
    void reset() {
        auto* begin = reinterpret_cast<char*>(storage_.data());
        setp(begin, begin + (storage_.size() - 1) * sizeof(float) + 1);
    }
    std::size_t written() const noexcept { return static_cast<std::size_t>(pptr() - pbase()); }
private:
    std::vector<float>& storage_;
};

// 公开 SDK RegRegion 是唯一传输入口；每次 MMIO 保持既有异常释放保护。
class TraceRegisters final : public RegisterAccess {
public:
    TraceRegisters(const Device& device, pose180_pipeline::ReleaseGuard& activity)
        : device_(device), activity_(activity) {}
    std::uint32_t read32(std::uint32_t offset) override {
        check(offset);
        activity_.begin("npu_trace_register_read");
        const auto result = device_.defaultRegRegion().read(0x400b6000ULL + offset);
        activity_.complete();
        return static_cast<std::uint32_t>(result);
    }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        check(offset);
        activity_.begin("npu_trace_register_write");
        device_.defaultRegRegion().write(0x400b6000ULL + offset, value);
        activity_.complete();
    }
private:
    void check(std::uint32_t offset) const {
        require(std::this_thread::get_id() == owner_ && activity_.permitted(),
                "NPU trace access needs the idle SDK owner");
        require(offset <= 0x28 && (offset & 3U) == 0, "NPU trace register offset is invalid");
    }
    Device device_;
    pose180_pipeline::ReleaseGuard& activity_;
    const std::thread::id owner_ = std::this_thread::get_id();
};

InferenceCompletion applied_completion(const Session& session, const xir::Network& graph,
                                        std::size_t expected_hardops) {
    std::vector<std::int64_t> hardops, ports, actual_ports;
    for (const auto& operation : graph->ops) {
        if (operation.is<xir::HardOp>()) {
            hardops.push_back(operation->op_id);
        } else if (operation.is<xir::Input>() || operation.is<xir::Output>()) {
            ports.push_back(operation->op_id);
        } else {
            throw std::runtime_error("unexpected Host computation in generated YOLO graph");
        }
    }
    require(hardops.size() == expected_hardops && ports.size() == 2, "qualified YOLO26l FP16 graph identity differs");
    std::vector<HardOpBinding> groups;
    for (const auto& binding : session.backendBindings()) {
        if (binding.second.is<HostBackend>()) {
            const auto operation = graph.getOpById(binding.first);
            require(operation.is<xir::Input>() || operation.is<xir::Output>(), "actual Host compute binding");
            actual_ports.push_back(binding.first);
            continue;
        }
        require(binding.second.is<zg330::ZG330Backend>(), "unknown runtime backend binding");
        const auto metadata = binding.second.cast<zg330::ZG330Backend>()->forward_info;
        require(metadata.defined(), "missing public ZG forward metadata");
        const auto found = metadata->hardop_map.find(binding.first);
        require(found != metadata->hardop_map.end() && found->second.defined(), "missing fused HardOp metadata");
        const auto entry = found->second;
        const auto index = metadata->idx_map.find(binding.first);
        require(entry->net_hardop.defined() && entry->net_hardop->op_id == binding.first &&
                index != metadata->idx_map.end() && index->second == entry->sync_idx,
                "public fusion node or completion metadata disagrees");
        groups.push_back({binding.first, index->second.first, index->second.second,
                          std::vector<std::int64_t>(entry->merge_from.begin(), entry->merge_from.end())});
    }
    return InferenceCompletion(std::move(hardops), std::move(ports), std::move(groups), std::move(actual_ports), 1);
}

} // namespace

struct IcraftRuntime::Impl {
    const std::thread::id owner = std::this_thread::get_id();
    IcraftConfig config;
    // 矩形图的 Host Tensor 借用该数组。它在张量及会话释放后才析构，初始化完成后不扩容。
    std::vector<float> normalized;
    // 反向析构顺序：张量/会话先于 Device。异常时 ReleaseGuard 在任何成员析构前保留整个 Impl。
    xir::Network graph;
    Device device;
    Session session;
    Tensor input;
    std::vector<Tensor> forward_inputs, outputs;
    pose180_pipeline::ReleaseGuard activity;
    std::unique_ptr<detail::IcraftVideoIo> video_io;
    std::unique_ptr<VideoEngine> video;
    std::optional<InferenceCompletion> completion;
    std::vector<std::uint8_t> packed_rgb;
    std::vector<float> raw;
    TensorSink sink{raw};
    std::size_t last_output_bytes = 0;
    std::uint64_t npu_callbacks = 0, host_compute_callbacks = 0;
    FrameKey last_key{};
    std::string deployment_report;
    std::vector<detail::NpuCounterSample> counter_samples;
    unsigned counter_columns = 0;
    bool counter_trace_armed = false, counter_trace_truncated = false;
    std::unique_ptr<TraceRegisters> trace_registers;
    std::unique_ptr<NpuTraceController> bus_trace;
    bool bus_trace_armed = false, bus_trace_captured = false;
    std::uint64_t bus_start_begin = 0, bus_start_end = 0, bus_stop_begin = 0, bus_stop_end = 0;
    bool last_inference_complete = false;
    bool uses_host_input_buffer = false;
    bool initialization_attempted = false, ready = false, failed = false, native_closed = false;

    ~Impl() {
        if (std::this_thread::get_id() != owner) {
            activity.begin("wrong_owner_thread");
        }
        if (activity.permitted()) {
            try {
                if (!close_known_idle()) {
                    activity.begin("video_or_native_completion_unproved");
                }
            } catch (...) {
                activity.begin("native_teardown_exception");
            }
        }
        activity.retain_if_unproved([&](const char* stage) {
            if (config.on_release_blocked) {
                config.on_release_blocked(stage);
            }
        });
    }

    void assert_owner() const {
        require(owner == std::this_thread::get_id(), "SDK access outside its single owning thread");
    }

    bool close_known_idle() {
        assert_owner();
        if (native_closed) {
            return true;
        }
        if (!activity.permitted()) {
            return false;
        }
        if (video && video->must_retain_resources()) {
            return false;
        }
        if (video_io && !video_io->release_known_idle()) {
            return false;
        }
        video.reset();
        video_io.reset();
        ready = false;
        activity.begin("session_teardown");
        if (session.defined()) {
            session.setPostCallBack(Session::CallBackFunc{});
        }
        outputs.clear();
        forward_inputs.clear();
        input = Tensor{};
        // Session::Create 拥有后端 deinit；此处不再单独重复 deinit 或 free 其内部张量。
        session = Session{};
        graph = xir::Network{};
        activity.complete();
        if (device.defined()) {
            const auto remaining = device.defaultMemRegion().memManager().getAllMemChunk().size();
            if (remaining != 0) {
                activity.begin("device_allocations_remain_after_session_teardown");
                return false;
            }
            activity.begin("device_close");
            Device::Close(device);
            activity.complete();
        }
        native_closed = true;
        return true;
    }

    IcraftIdentity initialize(const IcraftConfig& value) {
        assert_owner();
        require(!initialization_attempted, "SDK runtime initialization is single-use");
        initialization_attempted = true;
        config = value;
        require(config.output_read == IcraftOutputRead::Sfb || config.output_read == IcraftOutputRead::PlainFp32,
                "unknown output transfer mode");
        require(config.ocm_strategy >= -1 && config.ocm_strategy <= 3, "unknown OCM allocation strategy");
        require(config.input_profile == IcraftInputProfile::Fp32Normalized ||
                config.input_profile == IcraftInputProfile::RgbUInt8, "unknown YOLO input profile");
        const bool byte_input = config.input_profile == IcraftInputProfile::RgbUInt8;
        const auto geometry = config.geometry;
        require(geometry.valid() && (is_square_model(geometry) ||
                (geometry.width == kRectangularModel.width && geometry.height == kRectangularModel.height &&
                 config.model_variant == IcraftModelVariant::RegisteredCandidate && !byte_input)),
                "unsupported registered model geometry");
        require(config.model_variant == IcraftModelVariant::Baseline ||
                config.model_variant == IcraftModelVariant::ConcatPrefix16 ||
                config.model_variant == IcraftModelVariant::PointwiseTile20 ||
                config.model_variant == IcraftModelVariant::PointwiseMatmul ||
                config.model_variant == IcraftModelVariant::PointwiseAspect160 ||
                config.model_variant == IcraftModelVariant::PointwiseLayout12 ||
                config.model_variant == IcraftModelVariant::PointwiseMixed12 ||
                config.model_variant == IcraftModelVariant::PointwiseShared12 ||
                config.model_variant == IcraftModelVariant::PointwiseFlow4 ||
                config.model_variant == IcraftModelVariant::PointwiseFlow5 ||
                config.model_variant == IcraftModelVariant::Int8Psa32 ||
                config.model_variant == IcraftModelVariant::RegisteredCandidate, "unknown YOLO model variant");
        require(config.model_variant == IcraftModelVariant::RegisteredCandidate ?
                (config.candidate_hardops > 0 && config.candidate_hardops <= 4096) : config.candidate_hardops == 0,
                "registered candidate graph count is absent or outside its finite bound");
        const bool candidate_model = config.model_variant != IcraftModelVariant::Baseline;
        require(!candidate_model || (!byte_input && config.ocm_strategy == -1),
                "candidate model requires FP32 input and default OCM policy");
        std::size_t expected_hardops = byte_input ? 401 : 399;
        switch (config.model_variant) {
        case IcraftModelVariant::ConcatPrefix16:
            expected_hardops = 383;
            break;
        case IcraftModelVariant::PointwiseTile20:
            expected_hardops = 414;
            break;
        case IcraftModelVariant::PointwiseMatmul:
            expected_hardops = 417;
            break;
        case IcraftModelVariant::PointwiseAspect160:
            expected_hardops = 411;
            break;
        case IcraftModelVariant::PointwiseLayout12:
            expected_hardops = 454;
            break;
        case IcraftModelVariant::PointwiseMixed12:
            expected_hardops = 452;
            break;
        case IcraftModelVariant::PointwiseShared12:
            expected_hardops = 448;
            break;
        case IcraftModelVariant::PointwiseFlow4:
        case IcraftModelVariant::PointwiseFlow5:
            expected_hardops = 450;
            break;
        case IcraftModelVariant::Int8Psa32:
            expected_hardops = 449;
            break;
        case IcraftModelVariant::RegisteredCandidate:
            expected_hardops = config.candidate_hardops;
            break;
        case IcraftModelVariant::Baseline:
            break;
        }
        require(std::filesystem::is_regular_file(config.graph) && std::filesystem::is_regular_file(config.raw),
                "qualified model files are absent");
        graph = xir::Network::CreateFromJsonFile(config.graph.string());
        require(graph->icraft_version == "v3.39.0" && !graph->swap_mode, "model SDK version or swap contract differs");
        const auto inputs = graph.inputs(), graph_outputs = graph.outputs();
        require(inputs.size() == 1 && graph_outputs.size() == 1 &&
                shape_is(inputs[0].tensorType(), {1, geometry.height, geometry.width, 3}) &&
                shape_is(graph_outputs[0].tensorType(), {1, kRawChannels,
                                                       static_cast<std::int64_t>(geometry.anchors())}),
                "YOLO input/output contract differs");
        require((byte_input ? inputs[0].tensorType()->element_dtype.isUInt8() :
                              inputs[0].tensorType()->element_dtype.isFP32()) &&
                graph_outputs[0].tensorType()->element_dtype.isFP32(), "YOLO input profile or output type differs");
        require(inputs[0].tensorType()->layout.toString() == xir::Layout("NHWC").toString() &&
                graph_outputs[0].tensorType()->layout.toString() == xir::Layout("**C").toString(),
                "YOLO logical tensor layout differs");
        if (config.output_read == IcraftOutputRead::PlainFp32) {
            detail::validate_plain_fp32_output(graph_outputs[0].tensorType(), geometry);
        }
        std::size_t hardops = 0, ports = 0;
        for (const auto& operation : graph->ops) {
            require(operation.is<xir::Input>() || operation.is<xir::Output>() || operation.is<xir::HardOp>(),
                    "generated YOLO graph contains unsupported Host computation");
            hardops += operation.is<xir::HardOp>();
            ports += operation.is<xir::Input>() || operation.is<xir::Output>();
        }
        require(hardops == expected_hardops && ports == 2, "selected YOLO26l graph structure differs");
        graph.lazyLoadParamsFromFile(config.raw.string());
        // 只分配所选输入表示的工作区；未知 profile/模型在任何设备调用前拒绝。
        raw.resize(geometry.raw_elements() + 1);
        if (byte_input) {
            packed_rgb.resize(geometry.rgb_elements());
        } else {
            normalized.resize(geometry.rgb_elements());
        }
        require(!startup_consumed.test_and_set(), "this process already consumed its SDK startup");
        activity.begin("device_open");
        device = Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
        activity.complete();
        device.setWaitTime(10000);
        activity.begin("device_identity");
        const auto version = device.version();
        activity.complete();
        const auto actual_device = version.count("device") ? version.at("device") : "<missing>";
        const auto actual_icore = version.count("icore") ? version.at("icore") : "<missing>";
        require(actual_device == "25122301" && actual_icore == "FMSHZGV3TECH-AID - 24160628",
                "device/icore identity differs from the qualified platform");
        activity.begin("session_create_apply");
        session = Session::Create<zg330::ZG330Backend, HostBackend>(graph.view(), {device, HostDevice::Default()});
        session.enableTimeProfile(false);
        detail::configure_ocm(session, config.ocm_strategy);
        session.apply();
        activity.complete();
        completion.emplace(applied_completion(session, graph, expected_hardops));
        if (config.diagnostics) {
            activity.begin("npu_deployment_diagnostics");
            const auto report = detail::describe_npu_deployment(session, device, config.ocm_strategy);
            activity.complete();
            deployment_report = report.dump();
            counter_columns = report.at("cols").get<unsigned>();
            require(counter_columns >= 1 && counter_columns <= 4, "invalid diagnostic column count");
            counter_samples.reserve(512);
        }
        session.setPostCallBack([this](const Session&, const xir::Operation& operation, const Backend& backend,
                                       std::vector<Tensor>&) {
            assert_owner();
            if (backend.is<zg330::ZG330Backend>()) {
                ++npu_callbacks;
            } else if (!operation.is<xir::Input>() && !operation.is<xir::Output>()) {
                ++host_compute_callbacks;
            }
        });
        uses_host_input_buffer = !is_square_model(geometry);
        if (uses_host_input_buffer) {
            input = detail::wrap_host_fp32_input(inputs[0].tensorType(), normalized.data(), normalized.size(), geometry);
        } else {
            input = Tensor(inputs[0]).mallocOn(HostDevice::MemRegion());
        }
        forward_inputs.push_back(input);
        ready = true;
        return {actual_device, actual_icore, completion->final_count(), completion->groups(),
                byte_input ? "rgb_u8" : "fp32", geometry.rgb_elements() * (byte_input ? 1 : sizeof(float)),
                uses_host_input_buffer ? "host_buffer" : "sdk_write"};
    }

    InferenceResult infer(FrameKey key, std::uint64_t captured_ns, const std::uint8_t* rgb,
                           std::size_t bytes, std::size_t stride, float threshold) {
        assert_owner();
        require(ready && !failed && activity.permitted(), "SDK runtime is not ready or failed previously");
        const auto start = now_ns();
        last_inference_complete = false;
        last_output_bytes = 0;
        const bool trace_counters = counter_trace_armed;
        counter_trace_armed = false;
        counter_trace_truncated = false;
        counter_samples.clear();
        const bool trace_bus = bus_trace_armed;
        bus_trace_armed = false;
        bus_trace_captured = false;
        require(key.epoch && captured_ns && captured_ns <= start &&
                (!last_key.epoch || (key.epoch == last_key.epoch && key.sequence > last_key.sequence)) &&
                std::isfinite(threshold) && threshold >= 0 && threshold <= 1,
                "inference identity, capture time or threshold is invalid");
        sink.reset();
        InferenceResult result{key, captured_ns, 0, 0, 0, 0, {}, {}};
        const bool byte_input = config.input_profile == IcraftInputProfile::RgbUInt8;
        const auto geometry = config.geometry;
        if (byte_input) {
            pack_model_input_rgb(rgb, bytes, stride, packed_rgb.data(), packed_rgb.size(), geometry);
        } else {
            normalize_model_input(rgb, bytes, stride, normalized.data(), normalized.size(), geometry);
        }
        result.times.normalize_ns = now_ns() - start;
        last_key = key;
        if (!uses_host_input_buffer) {
            const auto begin = now_ns();
            if (byte_input) {
                input.write(0, reinterpret_cast<char*>(packed_rgb.data()), packed_rgb.size());
            } else {
                input.write(0, reinterpret_cast<char*>(normalized.data()), normalized.size() * sizeof(float));
            }
            result.times.input_write_ns = now_ns() - begin;
        }
        auto begin = now_ns();
        activity.begin("status_only_clear");
        device.reset(1);
        require(device.cast<ZG330Device>().layerCount() == 0, "status-only clear did not reach zero");
        activity.complete();
        result.times.clear_ns = now_ns() - begin;
        outputs.clear(); // 前一次调用已同时满足原生 wait 和实际融合计数。
        npu_callbacks = host_compute_callbacks = 0;
        if (trace_bus) {
            bus_start_begin = now_ns();
            // 按实际容量保持 102.4ms 窗口；16KiB 实现为 0.4ms/箱。此频率来自既有约束。
            const auto bins = bus_trace->capacity();
            bus_trace->start(20480000 / bins, bins);
            bus_start_end = now_ns();
        }
        begin = now_ns();
        activity.begin("forward_and_native_wait");
        if (trace_counters) {
            counter_samples.push_back(detail::sample_npu_counters(device, counter_columns));
        }
        outputs = session.forward(forward_inputs);
        result.times.forward_ns = now_ns() - begin;
        begin = now_ns();
        CompletionObservation observation{{}, 0, 0, 0};
        require(outputs.size() == 1, "actual YOLO output count differs");
        if (trace_counters) {
            // 单一 SDK 所有者采样，最多 512 行/500ms；采样结束后仍必须原生 wait 和计数校验。
            // 达到容量或时间界限只截断观察，绝不取消推理或释放在途资源。
            const auto deadline = Clock::now() + std::chrono::milliseconds(500);
            while (counter_samples.size() < 511 && Clock::now() < deadline) {
                counter_samples.push_back(detail::sample_npu_counters(device, counter_columns));
                if (counter_samples.back().layer_after >= completion->final_count()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            counter_trace_truncated = counter_samples.back().layer_after < completion->final_count();
        }
        for (const auto& tensor : outputs) {
            const bool waited = tensor.waitForReady(std::chrono::milliseconds(10000));
            observation.output_ready.push_back(waited && tensor.isReady());
        }
        observation.completed_layers = device.cast<ZG330Device>().layerCount();
        observation.npu_callbacks = npu_callbacks;
        observation.host_compute_callbacks = host_compute_callbacks;
        completion->validate(observation);
        if (trace_counters) {
            counter_samples.push_back(detail::sample_npu_counters(device, counter_columns));
        }
        activity.complete();
        result.times.final_wait_ns = now_ns() - begin;
        if (trace_bus) {
            bus_stop_begin = now_ns();
            bus_trace->stop();
            bus_stop_end = now_ns();
            bus_trace_captured = true;
        }
        result.completed_layers = observation.completed_layers;
        result.npu_callbacks = npu_callbacks;
        result.host_compute_callbacks = host_compute_callbacks;
        begin = now_ns();
        const auto raw_elements = geometry.raw_elements();
        if (config.output_read == IcraftOutputRead::PlainFp32) {
            activity.begin("first_plain_fp32_output_transfer");
            detail::read_plain_fp32_output(outputs[0], raw.data(), raw_elements, geometry);
            activity.complete();
        } else {
            std::ostream stream(&sink);
            activity.begin("first_sfb_output_transfer");
            outputs[0].dump(stream, "SFB");
            activity.complete();
            require(bool(stream) && sink.written() == raw_elements * sizeof(float), "native SFB output length differs");
        }
        last_output_bytes = raw_elements * sizeof(float);
        result.times.output_read_ns = now_ns() - begin;
        begin = now_ns();
        result.detections = decode_one_to_one(raw.data(), raw_elements, threshold, 300, geometry);
        result.times.postprocess_ns = now_ns() - begin;
        result.completed_ns = now_ns();
        last_inference_complete = true;
        return result;
    }
};

IcraftRuntime::IcraftRuntime() : impl_(std::make_unique<Impl>()) {}
IcraftRuntime::~IcraftRuntime() = default;

IcraftIdentity IcraftRuntime::initialize(const IcraftConfig& config) {
    require(bool(impl_), "SDK runtime already closed");
    try {
        return impl_->initialize(config);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

ModelGeometry IcraftRuntime::model_geometry() const {
    require(impl_ && impl_->ready, "SDK runtime has no initialized model");
    impl_->assert_owner();
    return impl_->config.geometry;
}

VideoEngine& IcraftRuntime::enable_video() {
    require(bool(impl_), "SDK runtime already closed");
    impl_->assert_owner();
    require(impl_->ready && !impl_->failed && impl_->activity.permitted() && !impl_->video_io,
            "video transport is unavailable or already initialized");
    const auto geometry = model_geometry();
    try {
        // 两阶段构造确保 malloc/初始化异常时 Impl 已持有适配器，不在栈展开中丢失在途内存。
        impl_->video_io = std::make_unique<detail::IcraftVideoIo>(impl_->device, impl_->activity);
        impl_->video_io->initialize(geometry);
        auto& io = *impl_->video_io;
        impl_->video = std::make_unique<VideoEngine>(io.preprocess_registers, io.display_registers,
                                                   io, io.allocations(), geometry);
        impl_->video->initialize();
        return *impl_->video;
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

InferenceResult IcraftRuntime::infer_rgb(FrameKey key, std::uint64_t captured_ns, const std::uint8_t* rgb,
                                        std::size_t bytes, std::size_t stride, float threshold) {
    require(bool(impl_), "SDK runtime already closed");
    try {
        return impl_->infer(key, captured_ns, rgb, bytes, stride, threshold);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

bool IcraftRuntime::close() {
    if (!impl_) {
        return true;
    }
    if (!impl_->close_known_idle()) {
        return false;
    }
    impl_.reset();
    return true;
}

bool IcraftRuntime::sdk_operations_complete() const noexcept { return !impl_ || impl_->activity.permitted(); }
bool IcraftRuntime::release_proven() const noexcept {
    return sdk_operations_complete() && (!impl_ || !impl_->video || !impl_->video->must_retain_resources());
}
const char* IcraftRuntime::pending_stage() const noexcept {
    if (!impl_) {
        return "closed";
    }
    if (impl_->activity.permitted() && impl_->video && impl_->video->must_retain_resources()) {
        return "video_dma_completion_unproved";
    }
    return impl_->activity.stage();
}
RawOutputView IcraftRuntime::last_output() const noexcept {
    return impl_ ? RawOutputView{reinterpret_cast<const char*>(impl_->raw.data()), impl_->last_output_bytes, impl_->last_key} :
                   RawOutputView{nullptr, 0, {}};
}

std::string IcraftRuntime::deployment_diagnostics() const {
    require(bool(impl_), "SDK runtime already closed");
    impl_->assert_owner();
    require(impl_->ready && impl_->config.diagnostics, "deployment diagnostics were not enabled");
    return impl_->deployment_report;
}

std::string IcraftRuntime::collect_npu_diagnostics() {
    require(bool(impl_), "SDK runtime already closed");
    impl_->assert_owner();
    require(impl_->ready && !impl_->failed && impl_->activity.permitted() && impl_->last_inference_complete &&
            impl_->config.diagnostics, "NPU diagnostics require a successfully completed inference");
    const auto begin = now_ns();
    impl_->activity.begin("completed_npu_timer_read");
    auto result = detail::read_npu_timers(impl_->session, impl_->device, impl_->completion->final_count());
    impl_->activity.complete();
    result["collection_ns"] = now_ns() - begin;
    result["epoch"] = impl_->last_key.epoch;
    result["frame"] = impl_->last_key.sequence;
    if (!impl_->counter_samples.empty()) {
        result["counter_timeline"] = detail::counter_timeline_json(
            impl_->counter_samples, impl_->counter_columns, impl_->counter_trace_truncated);
    }
    if (impl_->bus_trace_captured) {
        const auto read_begin = now_ns();
        const auto capture = impl_->bus_trace->read_frozen();
        result["bus_timeline"] = {{"schema", 1}, {"clock_period_ns_from_constraints", 5},
            {"period_cycles", capture.period_cycles}, {"capacity", capture.capacity},
            {"status", capture.status}, {"records", capture.records},
            {"arm_begin_ns", impl_->bus_start_begin}, {"arm_end_ns", impl_->bus_start_end},
            {"stop_begin_ns", impl_->bus_stop_begin}, {"stop_end_ns", impl_->bus_stop_end},
            {"readback_ns", now_ns() - read_begin}, {"mpe_wait_cycles", nullptr},
            {"actual_ddr_bytes", nullptr}};
    }
    return result.dump();
}

void IcraftRuntime::arm_counter_trace() {
    require(bool(impl_), "SDK runtime already closed");
    impl_->assert_owner();
    require(impl_->ready && !impl_->failed && impl_->activity.permitted() && impl_->config.diagnostics &&
            !impl_->counter_trace_armed, "counter tracing requires an idle diagnostic runtime");
    impl_->counter_trace_armed = true;
}

void IcraftRuntime::arm_bus_trace() {
    require(bool(impl_), "SDK runtime already closed");
    impl_->assert_owner();
    require(impl_->ready && !impl_->failed && impl_->activity.permitted() && impl_->config.diagnostics &&
            impl_->config.bus_trace && !impl_->bus_trace_armed, "bus tracing needs its explicit diagnostic permit");
    if (!impl_->bus_trace) {
        impl_->trace_registers = std::make_unique<TraceRegisters>(impl_->device, impl_->activity);
        impl_->bus_trace = std::make_unique<NpuTraceController>(*impl_->trace_registers);
        impl_->bus_trace->initialize();
    }
    impl_->bus_trace_armed = true;
}

} // namespace yolo26
