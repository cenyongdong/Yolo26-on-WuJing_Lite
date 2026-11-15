#include "yolo26/display_buffers.hpp"
#include "yolo26/compose.hpp"
#include <limits>
#include <stdexcept>

namespace yolo26 {

DisplayBuffers::DisplayBuffers(std::array<DmaRange, 3> memory) {
    constexpr std::uint64_t limit = std::uint64_t{1} << 32;
    for (std::size_t index = 0; index < slots_.size(); ++index) {
        const auto range = memory[index];
        if (range.address >= limit || (range.address & 63) || range.capacity < Compositor720p::kBytes ||
            range.capacity > limit - range.address) {
            throw std::invalid_argument("invalid owned 720p display allocation");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (range.address < memory[prior].address + memory[prior].capacity &&
                memory[prior].address < range.address + range.capacity) {
                throw std::invalid_argument("display allocations overlap");
            }
        }
        slots_[index].memory = range;
    }
}

const DisplayBuffers::Slot& DisplayBuffers::check_upload(const DisplayUpload& upload) const {
    if (upload.slot >= slots_.size()) {
        throw std::invalid_argument("unknown display upload slot");
    }
    const auto& slot = slots_[upload.slot];
    if (slot.state != State::Uploading || slot.generation != upload.generation || !(slot.frame == upload.frame)) {
        throw std::logic_error("stale or completed display upload lease");
    }
    return slot;
}

std::optional<DisplayUpload> DisplayBuffers::begin_upload(FrameKey key) {
    if (quarantined_ || stopping_ || stopped_ || !key.epoch ||
        (have_frame_ && (key.epoch != epoch_ || key.sequence <= last_sequence_))) {
        throw std::logic_error("display stopped, uncertain or frame is not new");
    }
    for (const auto& slot : slots_) {
        if (slot.state == State::Uploading) {
            return std::nullopt;
        }
    }
    for (std::size_t index = 0; index < slots_.size(); ++index) {
        auto& slot = slots_[index];
        if (slot.state != State::Free) {
            continue;
        }
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("display upload generation exhausted");
        }
        slot.state = State::Uploading;
        slot.generation = ++generation_;
        slot.frame = key;
        epoch_ = key.epoch;
        last_sequence_ = key.sequence;
        have_frame_ = true;
        return DisplayUpload{index, slot.generation, key};
    }
    return std::nullopt;
}

void DisplayBuffers::finish_upload(const DisplayUpload& upload) {
    check_upload(upload);
    if (quarantined_ || stopping_ || stopped_) {
        throw std::logic_error("cannot publish upload after stop or unknown transfer");
    }
    // 只替换尚未提交给硬件的 READY 完整画面；Submitted/Scanning 保留独占。
    for (auto& slot : slots_) {
        if (slot.state == State::Ready) {
            slot.state = State::Free;
            ++replaced_ready_;
        }
    }
    slots_[upload.slot].state = State::Ready;
}

void DisplayBuffers::abandon_upload(const DisplayUpload& upload) {
    check_upload(upload);
    if (quarantined_) {
        throw std::logic_error("unknown SDK transfer cannot release its upload lease");
    }
    // 仅用于同步上传从未开始或已正常返回的情况；SDK 异常必须调用 quarantine。
    slots_[upload.slot].state = State::Free;
}

std::uint32_t DisplayBuffers::address(const DisplayUpload& upload) const {
    return static_cast<std::uint32_t>(check_upload(upload).memory.address);
}

std::optional<DisplaySubmission> DisplayBuffers::begin_submit() {
    if (quarantined_ || stopping_ || stopped_) {
        throw std::logic_error("display cannot submit in its current state");
    }
    for (const auto& slot : slots_) {
        if (slot.state == State::Submitted) {
            return std::nullopt;
        }
    }
    for (std::size_t index = 0; index < slots_.size(); ++index) {
        auto& slot = slots_[index];
        if (slot.state != State::Ready) {
            continue;
        }
        // tag 0 保留给未采用状态；本次运行不允许回绕以免 ABA。
        if (next_id_ == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("display hardware tag exhausted; stop before restart");
        }
        slot.hardware_id = next_id_++;
        slot.state = State::Submitted;
        const bool start = !active_;
        active_ = true; // 在 MMIO 前置位：写已生效但抛异常时仍保留所有地址。
        return DisplaySubmission{index, slot.hardware_id, static_cast<std::uint32_t>(slot.memory.address),
                                  slot.frame, start};
    }
    return std::nullopt;
}

std::optional<FrameKey> DisplayBuffers::observe(const DisplaySnapshot& snapshot) {
    if (quarantined_ || stopping_ || stopped_) {
        throw std::logic_error("normal display observation is unavailable");
    }
    if (snapshot.faults || (snapshot.status & 0x80000000U) || !(snapshot.status & 0x40U) ||
        (active_ && (!(snapshot.status & 2U) || (snapshot.status & 0x81U)))) {
        quarantined_ = true;
        throw std::runtime_error("display stopped unexpectedly, rejected command or lost clock");
    }
    Slot* scanning = nullptr;
    Slot* submitted = nullptr;
    for (auto& slot : slots_) {
        if (slot.state == State::Scanning) {
            scanning = &slot;
        } else if (slot.state == State::Submitted) {
            submitted = &slot;
        }
    }
    if (submitted && snapshot.accepted_id == submitted->hardware_id &&
        snapshot.accepted_address == submitted->memory.address && (snapshot.status & 0x100U)) {
        // RTL 只有旧帧读事务结束后的帧边界才更新 accepted；邮箱命令应答本身不是这个证明。
        if (scanning) {
            scanning->state = State::Free;
        }
        submitted->state = State::Scanning;
        return submitted->frame;
    }
    const bool old_frame = scanning && snapshot.accepted_id == scanning->hardware_id &&
                           snapshot.accepted_address == scanning->memory.address && (snapshot.status & 0x100U);
    const bool not_started = !scanning && snapshot.accepted_id == 0 && snapshot.accepted_address == 0 &&
                             !(snapshot.status & 0x100U);
    if (!old_frame && !not_started) {
        quarantined_ = true;
        throw std::runtime_error("display adopted an unknown frame or address");
    }
    return std::nullopt;
}

std::uint32_t DisplayBuffers::begin_stop() {
    if (stopped_ || stopping_) {
        throw std::logic_error("display STOP already issued; poll the original command");
    }
    if (next_id_ == std::numeric_limits<std::uint32_t>::max()) {
        // 留给最终 STOP 的最后一个唯一 ID，不能继续启动新图。
        stop_id_ = next_id_;
    } else {
        stop_id_ = next_id_++;
    }
    stopping_ = true;
    return stop_id_;
}

bool DisplayBuffers::observe_stop(const DisplaySnapshot& snapshot) {
    if (!stopping_ || stopped_) {
        throw std::logic_error("no outstanding display STOP");
    }
    if ((snapshot.status & 0x80000000U) || snapshot.stopped_id != stop_id_ ||
        !(snapshot.status & 1U) || (snapshot.status & 0x3eU)) {
        return false;
    }
    // 扫描停止仅证明 HDMI 不再读内存；未知 SDK 上传仍必须单独保持租约。
    if (sdk_transfer_uncertain_) {
        return false;
    }
    for (const auto& slot : slots_) {
        if (slot.state == State::Uploading) {
            return false;
        }
    }
    for (auto& slot : slots_) {
        slot.state = State::Free;
    }
    active_ = false;
    stopped_ = true;
    quarantined_ = false;
    return true;
}

void DisplayBuffers::quarantine() noexcept {
    quarantined_ = true;
    sdk_transfer_uncertain_ = true;
}

std::size_t DisplayBuffers::occupied() const noexcept {
    std::size_t count = 0;
    for (const auto& slot : slots_) {
        count += slot.state != State::Free;
    }
    return count;
}

bool DisplayBuffers::must_retain_resources() const noexcept {
    return quarantined_ || active_ || (stopping_ && !stopped_) || occupied() != 0;
}

} // namespace yolo26
