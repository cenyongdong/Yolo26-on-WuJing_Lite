#pragma once
#include "yolo26/control.hpp"
#include "yolo26/display_control.hpp"
#include <array>
#include <thread>

namespace yolo26 {

// 启动时固定分配七块 PS DDR：0 为原图暂存，1..3 为模型输入，4..6 为显示三缓冲。
// 普通 Host 原图池仍为六槽；此暂存不把 V4L2 地址当作硬件地址。
using VideoAllocations = std::array<DmaRange, 7>;
class VideoMemoryAccess {
public:
    virtual ~VideoMemoryAccess() = default;
    virtual void write(std::size_t slot, const std::uint8_t* source, std::size_t bytes) = 0;
    virtual void read(std::size_t slot, std::uint8_t* destination, std::size_t bytes) = 0;
};

enum class PreprocessDisposition { Ready, NoHostBuffer, Stopped };
struct PreparedFrame {
    FrameRef original;
    FrameRef rgb;
    Letterbox geometry;
    ColorMatrix matrix;
    PreprocessDisposition disposition;
};

struct DisplayTiming {
    FrameInfo frame;
    std::uint64_t upload_started_ns;
    std::size_t slot;
    DisplaySnapshot snapshot;
};

// 单一 SDK 线程的非阻塞调度。只组合既有控制器/所有权账本，不复制 DMA 或像素算法。
// 分配对象和本对象都由外层保留，直到 must_retain_resources()==false；析构不做设备访问。
// SDK 传输异常时还保留正在读写的 Host 帧；外层不能单靠 HDMI STOP 就释放 SDK 输入指针。
class VideoEngine {
public:
    VideoEngine(RegisterAccess& preprocess_registers, RegisterAccess& display_registers,
                VideoMemoryAccess& memory, VideoAllocations allocations, ModelGeometry model = kSquareModel);
    ModelGeometry model_geometry() const noexcept { return model_; }
    void initialize();
    bool begin_preprocess(FrameRef frame, ColorMatrix matrix);
    std::optional<PreparedFrame> poll_preprocess(FramePool& model_pool);
    bool upload_display(FrameRef rgb565);
    std::optional<FrameKey> pump_display();
    // 仅由实际帧边界采用更新，用于把显示等待时长绑定回原始采集帧。
    const std::optional<DisplayTiming>& last_adoption() const noexcept { return last_adoption_; }
    std::vector<std::uint8_t> read_adopted_pixels(FrameKey frame);
    void request_stop();
    bool preprocess_busy() const noexcept { return bool(source_); }
    bool stopping() const noexcept { return stopping_; }
    bool must_retain_resources() const noexcept;
    std::uint64_t replaced_display_frames() const noexcept { return display_buffers_.replaced_ready(); }

private:
    void assert_owner() const;
    void require_new_work() const;
    PreparedFrame finish_preprocess(FrameRef rgb, PreprocessDisposition disposition);
    RegisterAccess& preprocess_registers_;
    VideoMemoryAccess& memory_;
    const ModelGeometry model_;
    const VideoAllocations allocations_;
    PreprocessController preprocess_;
    DisplayController display_;
    DisplayBuffers display_buffers_;
    const std::thread::id owner_ = std::this_thread::get_id();
    FrameRef source_, uploading_display_;
    FrameKey last_source_{};
    std::optional<FrameWriter> reading_model_;
    std::optional<PreprocessPlan> plan_;
    std::optional<Letterbox> configured_;
    std::optional<DisplaySubmission> submission_;
    std::array<DisplayTiming, 3> display_timing_{};
    std::optional<DisplayTiming> last_adoption_;
    std::vector<std::uint8_t> reading_display_;
    std::size_t input_slot_ = 1;
    std::uint32_t stop_tag_ = 0;
    bool initialized_ = false, stopping_ = false, stop_sent_ = false;
    bool transfer_uncertain_ = false, failed_ = false;
};

} // namespace yolo26
