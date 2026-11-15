#pragma once
#include "yolo26/frames.hpp"
#include "yolo26/preprocess.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace yolo26 {

struct CaptureConfig {
    std::string device = "/dev/video1";
    std::uint32_t width = 640;
    std::uint32_t height = 480;
    std::uint32_t fps = 30;
    std::uint64_t epoch = 0;
    std::string expected_bus_info;
};

struct CaptureFormat {
    std::uint32_t width, height, stride, size_image, buffers;
    std::uint32_t frame_interval_numerator, frame_interval_denominator;
    std::uint32_t colorspace, ycbcr_encoding, quantization, transfer_function;
    bool extended_fields_valid = false;
};

struct CaptureColor {
    ColorMatrix matrix;
    bool assumed_missing_metadata;
    std::uint32_t ycbcr_encoding, quantization, transfer_function;
};

// 仅解析已经协商的元数据，不操作设备。驱动未说明色彩时显式记录应用假设；不由 HDMI 分辨率推断。
// 当前 PL 仅支持 BT.601/709 的限幅/全幅 SDR 变换；其它编码或传递函数拒绝而不静默近似。
CaptureColor resolve_capture_color(const CaptureFormat& format, ColorMatrix missing_metadata_fallback);

struct CaptureCounters {
    std::uint64_t dequeued = 0, published = 0, driver_sequence_gaps = 0;
    std::uint64_t duplicate_or_reordered = 0, flagged_error = 0, invalid_bytes = 0;
    std::uint64_t invalid_timestamp = 0, pool_exhausted = 0, latest_replaced = 0, poll_timeouts = 0;
    std::uint64_t driver_timestamps = 0, receive_timestamps = 0;
};

// 单线程所有者的 Linux UVC YUYV 采集器。必须由外层持板卡/摄像头租约后创建。
// 构造只协商和分配，start 才启流。next 将最新完整帧复制到普通 Host 池后立即归还 V4L2 缓冲。
// 不把 V4L2 MMAP 地址交给 FPGA，不依赖用户指针或未证实的 DMABUF/SDK 互操作。
class CameraCapture {
public:
    explicit CameraCapture(const CaptureConfig& config);
    ~CameraCapture();
    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;
    const CaptureFormat& format() const;
    const CaptureCounters& counters() const;
    void start();
    // timeout_ms 为 0..1000；超时/坏帧/满池返回空。系统调用硬错误关闭本次采集并抛出异常。
    FrameRef next(FramePool& pool, int timeout_ms = 100);
    // 显式停止并关闭，仅返回所有 V4L2 清理调用是否正常；与自研 PL/NPU 排空证明独立。
    bool stop() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::uint64_t monotonic_ns();

} // namespace yolo26
