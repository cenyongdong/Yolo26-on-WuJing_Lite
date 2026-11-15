#pragma once
#include "yolo26/frames.hpp"
#include "yolo26/preprocess.hpp"
#include <array>
#include <cstdint>
#include <optional>

namespace yolo26 {

struct DisplayUpload {
    std::size_t slot;
    std::uint64_t generation;
    FrameKey frame;
};

struct DisplaySubmission {
    std::size_t slot;
    std::uint32_t hardware_id;
    std::uint32_t address;
    FrameKey frame;
    bool start;
};

struct DisplaySnapshot {
    std::uint32_t status, stopped_id, accepted_address, accepted_id, faults;
};

// 仅管理三块已分配显示 DMA 内存的所有权；不分配、释放、映射内存或操作寄存器。
// 单一 SDK 所有者调用，上传/提交/扫描互斥。返回空时保留管线的最新待处理画面。
class DisplayBuffers {
public:
    explicit DisplayBuffers(std::array<DmaRange, 3> memory);
    std::optional<DisplayUpload> begin_upload(FrameKey key);
    void finish_upload(const DisplayUpload& upload);
    void abandon_upload(const DisplayUpload& upload);
    std::optional<DisplaySubmission> begin_submit();
    std::optional<FrameKey> observe(const DisplaySnapshot& snapshot);
    std::uint32_t begin_stop();
    bool observe_stop(const DisplaySnapshot& snapshot);
    void quarantine() noexcept;

    std::uint32_t address(const DisplayUpload& upload) const;
    std::size_t occupied() const noexcept;
    bool must_retain_resources() const noexcept;
    bool stopped() const noexcept { return stopped_; }
    std::uint64_t replaced_ready() const noexcept { return replaced_ready_; }

private:
    enum class State { Free, Uploading, Ready, Submitted, Scanning };
    struct Slot {
        DmaRange memory;
        State state = State::Free;
        FrameKey frame{};
        std::uint64_t generation = 0;
        std::uint32_t hardware_id = 0;
    };
    const Slot& check_upload(const DisplayUpload& upload) const;
    std::array<Slot, 3> slots_;
    std::uint64_t generation_ = 0, epoch_ = 0, last_sequence_ = 0, replaced_ready_ = 0;
    std::uint32_t next_id_ = 1, stop_id_ = 0;
    bool have_frame_ = false, active_ = false, stopping_ = false, stopped_ = false, quarantined_ = false;
    bool sdk_transfer_uncertain_ = false;
};

} // namespace yolo26
