#include "yolo26/npu_trace.hpp"
#include <chrono>
#include <stdexcept>

namespace yolo26 {
namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

constexpr std::uint32_t kBusy = 0x18, kResult = 0x1c, kIssued = 0x20, kCompleted = 0x24;
constexpr unsigned kMaxPolls = 4096;
using Clock = std::chrono::steady_clock;

} // namespace

void NpuTraceController::require_idle_mailbox() {
    require(!failed_, "NPU trace controller is quarantined");
    require(registers_.read32(kBusy) == 0 && registers_.read32(kIssued) == serial_ &&
            registers_.read32(kCompleted) == serial_, "NPU trace mailbox is not exclusively idle");
}

std::uint32_t NpuTraceController::command(std::uint32_t opcode) {
    try {
        require_idle_mailbox();
        registers_.write32(0x10, opcode);
        ++serial_; // uint32_t 回绕与硬件完全一致。
        const auto deadline = Clock::now() + std::chrono::milliseconds(50);
        for (unsigned poll = 0; poll < kMaxPolls && Clock::now() < deadline; ++poll) {
            const auto busy = registers_.read32(kBusy);
            require(busy <= 1 && registers_.read32(kIssued) == serial_, "NPU trace serial changed externally");
            if (!busy && registers_.read32(kCompleted) == serial_) {
                return registers_.read32(kResult);
            }
        }
        throw std::runtime_error("NPU trace mailbox timed out; command may remain pending");
    } catch (...) {
        failed_ = true;
        throw;
    }
}

void NpuTraceController::check_status(std::uint32_t status) const {
    require((status & 0xfff0U) == 0 && (status >> 16) <= depth_ && (status & 8U) == 0,
            "NPU trace returned an invalid status or command error");
}

void NpuTraceController::initialize() {
    require(!initialized_ && !failed_, "NPU trace initialization is single-use");
    try {
        require(registers_.read32(0) == 0x4e505431 && registers_.read32(4) == 0x00010000,
                "NPU trace bitstream identity/ABI differs");
        depth_ = registers_.read32(0x28);
        require(depth_ == 256 || depth_ == 512, "NPU trace capture capacity differs");
        serial_ = registers_.read32(kIssued);
        require_idle_mailbox();
        status_ = command(0);
        check_status(status_);
        require((status_ & 1U) == 0, "NPU trace is already active; do not take over another owner");
        initialized_ = true;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

void NpuTraceController::start(std::uint32_t period_cycles, std::uint32_t bins) {
    require(initialized_ && !failed_ && (!started_ || frozen_), "NPU trace cannot start in this state");
    require(period_cycles > 0 && period_cycles <= 1000000 && bins > 0 && bins <= depth_,
            "NPU trace period or bin count is outside its bounded capacity");
    try {
        require_idle_mailbox();
        registers_.write32(0x08, period_cycles);
        registers_.write32(0x0c, bins);
        status_ = command(1);
        check_status(status_);
        require((status_ & 3U) != 0 && (status_ & 4U) == 0, "NPU trace did not start");
        period_ = period_cycles;
        limit_ = bins;
        started_ = true;
        frozen_ = false;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

void NpuTraceController::stop() {
    require(started_ && !frozen_ && !failed_, "NPU trace is not available for stop");
    try {
        status_ = command(2);
        for (unsigned poll = 0; poll < 8; ++poll) {
            check_status(status_);
            if ((status_ & 3U) == 2U) {
                require((status_ >> 16) <= limit_, "NPU trace captured more bins than requested");
                frozen_ = true;
                return;
            }
            status_ = command(0); // STOP 应答不代替最后一箱写入完成。
        }
        throw std::runtime_error("NPU trace final record has not committed");
    } catch (...) {
        failed_ = true;
        throw;
    }
}

NpuBusCapture NpuTraceController::read_frozen() {
    require(frozen_ && !failed_, "NPU trace records are not frozen");
    try {
        NpuBusCapture capture{period_, limit_, status_, {}};
        capture.records.resize(status_ >> 16);
        std::uint64_t previous_end = 0;
        for (std::size_t bin = 0; bin < capture.records.size(); ++bin) {
            auto& words = capture.records[bin];
            for (std::size_t word = 0; word < words.size(); ++word) {
                require_idle_mailbox();
                registers_.write32(0x14, static_cast<std::uint32_t>(bin * 16 + word));
                // 记录可以包含任意 32bit 值，不能把类似错误码的合法数据误当命令错误。
                words[word] = command(3);
            }
            const auto tick = (static_cast<std::uint64_t>(words[3]) << 32) | words[2];
            require(words[0] == bin && words[1] > 0 && words[1] <= period_ &&
                    (bin == 0 || tick == previous_end), "NPU trace record order or interval differs");
            for (unsigned field = 4; field <= 13; ++field) {
                if (field != 10) {
                    require(words[field] <= words[1], "NPU trace event count exceeds its bin");
                }
            }
            require(words[10] <= 64U * words[8], "NPU trace byte enables exceed accepted writes");
            previous_end = tick + words[1];
        }
        require(command(0) == status_, "NPU trace changed while reading its frozen records");
        return capture;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

} // namespace yolo26
