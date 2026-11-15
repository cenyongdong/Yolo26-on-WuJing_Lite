#pragma once
#include "yolo26/frames.hpp"
#include "yolo26/inference_completion.hpp"
#include "yolo26/postprocess.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace yolo26 {

class VideoEngine;

enum class IcraftInputProfile { Fp32Normalized, RgbUInt8 };
enum class IcraftOutputRead { Sfb, PlainFp32 };
enum class IcraftModelVariant {
    Baseline, ConcatPrefix16, PointwiseTile20, PointwiseMatmul, PointwiseAspect160, PointwiseLayout12,
    PointwiseMixed12, PointwiseShared12, PointwiseFlow4, PointwiseFlow5, Int8Psa32, RegisteredCandidate
};

struct IcraftConfig {
    std::filesystem::path graph, raw;
    // 未证明完成时由既有 ReleaseGuard 调用，然后保留进程/设备/输入输出/继承锁，不重试 SDK。
    std::function<void(const char*)> on_release_blocked;
    // 显式选择已经核验哈希和数值的输入图；默认仍是原 FP32 输入基线。
    IcraftInputProfile input_profile = IcraftInputProfile::Fp32Normalized;
    // -1 为 SDK BEST_SCORE；0 为关闭；1/2/3 为三种公开方案。仅在 apply 前配置。
    int ocm_strategy = -1;
    bool diagnostics = false;
    // 只有核验过 NPT1 位流的有限文件包才能启用；普通推理不访问新增 CSR。
    bool bus_trace = false;
    // 独立文件实验，模型身份由有限包固定；输入仍是 FP32，不改变默认模型。
    IcraftModelVariant model_variant = IcraftModelVariant::Baseline;
    // 仅 RegisteredCandidate 使用；来自核验固定 JSON/RAW 身份的包登记，不从任意模型自动放行。
    std::size_t candidate_hardops = 0;
    // 有限文件实验；PlainFp32 只接受无填充的公开 FP32 输出，仍通过 SDK read 完成传输。
    IcraftOutputRead output_read = IcraftOutputRead::Sfb;
    // 矩形图只供已登记的文件回放；板端实时 PL 预处理使用独立的固定几何。
    ModelGeometry geometry = kSquareModel;
};

struct IcraftIdentity {
    std::string device, icore;
    std::uint32_t completion_layers;
    std::vector<HardOpBinding> fusion;
    std::string input_profile;
    std::size_t logical_input_bytes;
    std::string input_transfer;
};

struct InferenceTimes {
    // normalize_ns 保留报告字段名；UInt8 profile 记录 RGB 去行填充/拷贝耗时，未做 CPU /255。
    std::uint64_t normalize_ns = 0, input_write_ns = 0, clear_ns = 0;
    std::uint64_t forward_ns = 0, final_wait_ns = 0, output_read_ns = 0, postprocess_ns = 0;
};

struct InferenceResult {
    FrameKey key;
    std::uint64_t captured_ns, completed_ns;
    std::uint32_t completed_layers;
    std::uint64_t npu_callbacks, host_compute_callbacks;
    InferenceTimes times;
    std::vector<Detection> detections;
};

struct RawOutputView {
    const char* bytes;
    std::size_t size;
    FrameKey key;
};

// 单个 SDK 所有者线程。构造只分配 Host 工作区；initialize 才消耗一次设备功能启动。
// 调用 initialize 前，板端包装器必须已持有共享锁、核验 BOOT/模型哈希并记录本轮启动。
// 所有者进程须安装 TERM/INT/HUP 的“请求停止”处理器，不能用信号/超时强制回收未知硬件事务。
// 首版在同一 SDK 线程做尾处理，返回的小型检测结果可交给独立合成线程。
class IcraftRuntime {
public:
    IcraftRuntime();
    ~IcraftRuntime();
    IcraftRuntime(const IcraftRuntime&) = delete;
    IcraftRuntime& operator=(const IcraftRuntime&) = delete;

    IcraftIdentity initialize(const IcraftConfig& config);
    ModelGeometry model_geometry() const;
    // 仅在已核验的 YOLO SoC BOOT 上启用；主动提交之前核对 PL 输出尺寸及 HDMI 时序。
    // 返回借用接口，到 close 为止有效，仍只能由同一 SDK 所有者线程访问。
    VideoEngine& enable_video();
    InferenceResult infer_rgb(FrameKey key, std::uint64_t captured_ns, const std::uint8_t* rgb,
                              std::size_t bytes, std::size_t stride, float threshold = 0.25F);
    // 仅在所有推理同步完成后关闭。不访问未知在途硬件；不满足时返回 false 并保留所有对象。
    bool close();
    // SDK 原生操作是否已完成；仅用于决定能否继续发出视频 STOP，不替代视频排空。
    bool sdk_operations_complete() const noexcept;
    bool release_proven() const noexcept;
    const char* pending_stage() const noexcept;
    // 只读首份 SFB 原值/失败前缀；生命周期到下一次 infer_rgb 或 close，不能跨帧借用。
    RawOutputView last_output() const noexcept;
    // 诊断快照不借出 SDK 对象。计时读取仅允许在 infer_rgb 已成功返回后进行，耗时单独记录。
    std::string deployment_diagnostics() const;
    std::string collect_npu_diagnostics();
    // 仅下一次已授权的文件诊断调用采样公开计数器；不改变普通性能样本或原生完成条件。
    void arm_counter_trace();
    void arm_bus_trace();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yolo26
