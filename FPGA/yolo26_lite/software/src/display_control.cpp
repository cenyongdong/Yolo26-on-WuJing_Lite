#include "yolo26/display_control.hpp"
#include <stdexcept>

namespace yolo26 {
namespace {

constexpr std::uint32_t kCoreAbi = 0x48444d01, kBoardAbi = 0x48445301;
constexpr std::uint32_t kWidthHeight = (720U << 16) | 1280U;
constexpr std::uint32_t kAddress = 0x04, kTag = 0x08, kCommand = 0x0c;
constexpr std::uint32_t kIssued = 0x24, kCompleted = 0x28, kMailbox = 0x2c;
constexpr std::uint32_t kAdapter = 0x30, kPattern = 0x34;

std::uint32_t opcode(DisplayCommand command) {
    switch (command) {
    case DisplayCommand::Initialize:
    case DisplayCommand::Snapshot:
        return 0;
    case DisplayCommand::Start:
        return 1;
    case DisplayCommand::Flip:
        return 2;
    case DisplayCommand::Stop:
        return 3;
    }
    throw std::logic_error("unknown display command");
}

} // namespace

DisplayController::DisplayController(RegisterAccess& registers) noexcept : registers_(registers) {}

std::uint32_t DisplayController::read(std::uint32_t offset) {
    try {
        return registers_.read32(offset);
    } catch (...) {
        uncertain_ = true;
        throw;
    }
}

void DisplayController::write(std::uint32_t offset, std::uint32_t value) {
    try {
        registers_.write32(offset, value);
    } catch (...) {
        // 写抛出异常不证明写没有生效；此控制器不再重发或清错。
        uncertain_ = true;
        throw;
    }
}

[[noreturn]] void DisplayController::quarantine(const char* message) {
    uncertain_ = true;
    throw std::runtime_error(message);
}

bool DisplayController::ready() const noexcept {
    return initialized_ && !pending_ && !uncertain_;
}

void DisplayController::initialize() {
    if (started_ || uncertain_) {
        throw std::logic_error("display controller initialization is single-use");
    }
    started_ = true;
    if (read(0) != kCoreAbi || read(0x40) != kBoardAbi || read(0x44) != 74250000 ||
        read(0x48) != 371250000 || read(0x4c) != kWidthHeight) {
        quarantine("720p display identity or clock contract differs");
    }
    completed_ = read(kCompleted);
    if (read(kIssued) != completed_ || read(kMailbox) != 0 || (read(kAdapter) & 9U) != 1U) {
        quarantine("display mailbox has an owner, error or missing clock");
    }
    if (!issue(DisplayCommand::Initialize, 0, 0)) {
        quarantine("display initialization mailbox changed unexpectedly");
    }
}

bool DisplayController::issue(DisplayCommand command, std::uint32_t address, std::uint32_t tag) {
    if (uncertain_ || !started_) {
        throw std::logic_error("display controller is uninitialized or uncertain");
    }
    if (pending_) {
        return false;
    }
    const auto mailbox = read(kMailbox);
    if ((mailbox & ~3U) || read(kCompleted) != completed_ || read(kIssued) != completed_) {
        quarantine("display mailbox ownership or generation changed");
    }
    if (mailbox != 0) {
        // 响应计数已更新，但跨域邮箱仍可能在返还 ready；本轮不写，下一轮再轮询。
        return false;
    }
    command_ = command;
    pending_ = true; // 在任意参数写入前记录不确定边界，防止异常后提前析构外部资源。
    write(kAddress, address);
    write(kTag, tag);
    write(kCommand, opcode(command));
    return true;
}

bool DisplayController::request_snapshot() {
    if (!initialized_) {
        throw std::logic_error("display initial snapshot is not verified");
    }
    return issue(DisplayCommand::Snapshot, 0, 0);
}

bool DisplayController::submit(const DisplaySubmission& submission) {
    if (!initialized_ || !submission.hardware_id || (submission.address & 63U)) {
        throw std::invalid_argument("display submission is not initialized, tagged and aligned");
    }
    // 调用者必须先在 DisplayBuffers.begin_submit() 中保留这块内存，再进行 MMIO。
    return issue(submission.start ? DisplayCommand::Start : DisplayCommand::Flip,
                 submission.address, submission.hardware_id);
}

bool DisplayController::request_stop(std::uint32_t tag) {
    if (!initialized_ || !tag) {
        throw std::invalid_argument("display STOP requires initialization and a unique nonzero tag");
    }
    return issue(DisplayCommand::Stop, 0, tag);
}

std::optional<DisplayReply> DisplayController::poll() {
    if (uncertain_) {
        throw std::logic_error("display MMIO outcome is uncertain; retain original resources");
    }
    if (!pending_) {
        return std::nullopt;
    }
    const auto response = read(kCompleted);
    if (read(kMailbox) & ~3U) {
        quarantine("display CSR error while waiting for response");
    }
    if (response == completed_) {
        return std::nullopt;
    }
    // CSR 计数器按 32bit 回绕；本实例始终最多一个在途命令，因此等值检查不产生 ABA。
    const auto expected = static_cast<std::uint32_t>(completed_ + 1U);
    if (response != expected || read(kIssued) != expected) {
        quarantine("display response generation skipped or belongs to another owner");
    }
    const DisplaySnapshot snapshot{read(0x10), read(0x14), read(0x18), read(0x1c), read(0x20)};
    if (read(kCompleted) != response || (read(kAdapter) & 8U)) {
        quarantine("display snapshot changed during read or adapter rejected a command");
    }
    completed_ = response;
    pending_ = false;
    if (command_ == DisplayCommand::Initialize) {
        if (!(snapshot.status & 1U) || !(snapshot.status & 0x40U) ||
            (snapshot.status & (0x80000000U | 0x13eU)) || snapshot.faults ||
            snapshot.stopped_id || snapshot.accepted_id || snapshot.accepted_address) {
            quarantine("display initialization is not a fresh, locked and drained startup");
        }
        // 只有已证明空闲才关闭静态测试图；它不触发 DMA START。
        write(kPattern, 0);
        if (read(kPattern) != 0 || (read(kAdapter) & 9U) != 1U) {
            quarantine("display static pattern did not disable safely");
        }
        initialized_ = true;
    }
    return DisplayReply{command_, snapshot, response};
}

} // namespace yolo26
