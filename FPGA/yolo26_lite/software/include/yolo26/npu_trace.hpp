#pragma once
#include "yolo26/control.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace yolo26 {

struct NpuBusCapture {
    std::uint32_t period_cycles = 0, capacity = 0, status = 0;
    std::vector<std::array<std::uint32_t, 16>> records;
};

// 单一所有者。只控制无 DDR 写主机的观察器，不控制或证明 NPU 完成。
// MMIO 失败或邮箱超时后本对象封锁；不重发不确定命令，析构不访问硬件。
class NpuTraceController {
public:
    explicit NpuTraceController(RegisterAccess& registers) : registers_(registers) {}
    void initialize();
    void start(std::uint32_t period_cycles, std::uint32_t bins);
    void stop();
    NpuBusCapture read_frozen();
    bool failed() const noexcept { return failed_; }
    std::uint32_t capacity() const noexcept { return depth_; }

private:
    std::uint32_t command(std::uint32_t opcode);
    void require_idle_mailbox();
    void check_status(std::uint32_t status) const;
    RegisterAccess& registers_;
    std::uint32_t serial_ = 0, depth_ = 0, period_ = 0, limit_ = 0, status_ = 0;
    bool initialized_ = false, started_ = false, frozen_ = false, failed_ = false;
};

} // namespace yolo26
