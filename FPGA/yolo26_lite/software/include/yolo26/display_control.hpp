#pragma once
#include "yolo26/control.hpp"
#include "yolo26/display_buffers.hpp"
#include <cstdint>
#include <optional>

namespace yolo26 {

enum class DisplayCommand { Initialize, Snapshot, Start, Flip, Stop };
struct DisplayReply {
    DisplayCommand command;
    DisplaySnapshot snapshot;
    std::uint32_t generation;
    bool accepted() const noexcept { return (snapshot.status & 0x80000000U) == 0; }
};

// 单一 SDK 线程使用的非阻塞 HDMI 邮箱。时间预算由外层负责，超时后继续保留资源。
// START/FLIP/STOP 应答是命令执行前的快照，不能交给 DisplayBuffers 当作采用/排空证明；
// 只用后续 SNAPSHOT 应答更新所有权。类不分配内存，也不在析构中操作硬件。
class DisplayController {
public:
    explicit DisplayController(RegisterAccess& registers) noexcept;
    DisplayController(const DisplayController&) = delete;
    DisplayController& operator=(const DisplayController&) = delete;
    void initialize();
    bool request_snapshot();
    bool submit(const DisplaySubmission& submission);
    bool request_stop(std::uint32_t tag);
    std::optional<DisplayReply> poll();
    bool ready() const noexcept;
    bool pending() const noexcept { return pending_; }
    bool must_retain_resources() const noexcept { return pending_ || uncertain_; }

private:
    std::uint32_t read(std::uint32_t offset);
    void write(std::uint32_t offset, std::uint32_t value);
    bool issue(DisplayCommand command, std::uint32_t address, std::uint32_t tag);
    [[noreturn]] void quarantine(const char* message);
    RegisterAccess& registers_;
    bool started_ = false, initialized_ = false, pending_ = false, uncertain_ = false;
    DisplayCommand command_ = DisplayCommand::Initialize;
    std::uint32_t completed_ = 0;
};

} // namespace yolo26
