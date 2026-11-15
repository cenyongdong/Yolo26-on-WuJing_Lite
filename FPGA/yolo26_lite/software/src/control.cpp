#include "yolo26/control.hpp"
#include <stdexcept>
#include <utility>

namespace yolo26 {
namespace {

constexpr std::uint32_t kAbi = 0x59323601;
constexpr std::uint32_t kCommand = 0x04, kState = 0x08, kErrors = 0x0c, kTag = 0x10;
constexpr std::uint32_t kStop = 0x3c, kResponse = 0x40, kAck = 0x5c;
constexpr std::uint32_t kIssued = 0x60, kCompleted = 0x64;
constexpr std::uint32_t kStart = 1, kMapClear = 2, kMapWrite = 3;
constexpr std::uint32_t kStopped = 5;

bool same_geometry(const Letterbox& a, const Letterbox& b) {
    return a.source_width == b.source_width && a.source_height == b.source_height &&
           a.target_width == b.target_width && a.target_height == b.target_height &&
           a.resized_width == b.resized_width && a.resized_height == b.resized_height &&
           a.left == b.left && a.top == b.top && a.gain == b.gain;
}

// PreprocessPlan 是可审阅的数据结构，因此在进入硬件前重新验证调用方没有修改其派生字段。
PreprocessPlan checked_plan(const PreprocessPlan& plan) {
    auto canonical = make_preprocess_plan(plan.source, plan.source_stride, plan.geometry.source_width,
                                          plan.geometry.source_height, plan.source_format, plan.matrix,
                                          plan.destination, plan.destination_stride,
                                          {plan.geometry.target_width, plan.geometry.target_height});
    if (!same_geometry(plan.geometry, canonical.geometry) || plan.horizontal_map != canonical.horizontal_map ||
        plan.vertical_map != canonical.vertical_map) {
        throw std::invalid_argument("preprocess plan contains inconsistent derived geometry or maps");
    }
    return canonical;
}

} // namespace

PreprocessController::PreprocessController(RegisterAccess& registers) noexcept : registers_(registers) {}

std::uint32_t PreprocessController::read(std::uint32_t offset) {
    try {
        return registers_.read32(offset);
    } catch (...) {
        phase_ = ControlPhase::Quarantined;
        throw;
    }
}

void PreprocessController::write(std::uint32_t offset, std::uint32_t value) {
    try {
        registers_.write32(offset, value);
    } catch (...) {
        // 写异常可能发生在硬件已经接收之后，不能自动重发或返回 Idle。
        phase_ = ControlPhase::Quarantined;
        throw;
    }
}

[[noreturn]] void PreprocessController::quarantine(const char* message) {
    phase_ = ControlPhase::Quarantined;
    throw std::runtime_error(message);
}

bool PreprocessController::must_retain_resources() const noexcept {
    return phase_ == ControlPhase::Configuring || phase_ == ControlPhase::Running ||
           phase_ == ControlPhase::Quarantined;
}

void PreprocessController::require_idle() const {
    if (phase_ != ControlPhase::Idle) {
        throw std::logic_error("preprocess controller is not idle");
    }
}

void PreprocessController::check_hardware_idle() {
    const auto state = read(kState);
    if ((state & ~0x10U) != 0 || read(kErrors) != 0 ||
        read(kIssued) != completed_count_ || read(kCompleted) != completed_count_) {
        quarantine("preprocess hardware is not exclusively idle");
    }
    if (bool(state & 0x10U) != stop_requested_) {
        quarantine("preprocess STOP changed outside its owner");
    }
}

void PreprocessController::initialize() {
    if (phase_ != ControlPhase::Uninitialized) {
        throw std::logic_error("controller initialization cannot recover unknown hardware state");
    }
    if (read(0) != kAbi) {
        quarantine("unsupported preprocess ABI");
    }
    completed_count_ = read(kCompleted);
    stop_requested_ = (read(kState) & 0x10U) != 0;
    check_hardware_idle();
    phase_ = ControlPhase::Idle;
}

void PreprocessController::issue(std::uint32_t opcode) {
    check_hardware_idle();
    pending_tag_ = next_tag_++;
    pending_opcode_ = opcode;
    write(kTag, pending_tag_);
    // 命令寄存器是最后一次写；此前所有字段只更新空闲的 GP 参数快照。
    write(kCommand, opcode);
}

void PreprocessController::configure(const PreprocessPlan& plan) {
    require_idle();
    if (stop_requested_) {
        throw std::logic_error("clear STOP before configuring preprocessing");
    }
    auto canonical = checked_plan(plan);
    check_hardware_idle();
    configuration_ = std::move(canonical);
    configured_geometry_.reset();
    configuration_step_ = 0;
    phase_ = ControlPhase::Configuring;
    issue(kMapClear);
}

void PreprocessController::issue_map_entry() {
    const auto& plan = *configuration_;
    const auto index = configuration_step_ - 1;
    const bool axis_y = index >= plan.horizontal_map.size();
    const auto axis_index = axis_y ? index - plan.horizontal_map.size() : index;
    write(0x34, (axis_y ? 0x400U : 0U) | static_cast<std::uint32_t>(axis_index));
    write(0x38, axis_y ? plan.vertical_map[axis_index] : plan.horizontal_map[axis_index]);
    issue(kMapWrite);
}

std::uint32_t PreprocessController::start(const PreprocessPlan& plan) {
    require_idle();
    if (stop_requested_) {
        throw std::logic_error("clear STOP before starting preprocessing");
    }
    const auto canonical = checked_plan(plan);
    if (!configured_geometry_ || !same_geometry(*configured_geometry_, canonical.geometry)) {
        throw std::logic_error("preprocess geometry does not match successfully uploaded maps");
    }
    check_hardware_idle();
    expected_source_bytes_ = canonical.geometry.source_width * canonical.geometry.source_height *
                             (canonical.source_format == PixelFormat::Rgb888 ? 3U : 2U);
    expected_destination_bytes_ = canonical.geometry.target_width * canonical.geometry.target_height * 3;
    // 首次副作用前就进入保留状态，任意中途异常都不能让调用方提前回收物理内存。
    phase_ = ControlPhase::Running;
    write(0x14, static_cast<std::uint32_t>(canonical.source.address));
    write(0x18, canonical.source_stride);
    write(0x1c, canonical.geometry.source_height << 16 | canonical.geometry.source_width);
    write(0x20, (canonical.source_format == PixelFormat::Rgb888 ? 1U : 0U) |
                static_cast<std::uint32_t>(canonical.matrix) << 1);
    write(0x24, static_cast<std::uint32_t>(canonical.destination.address));
    write(0x28, canonical.destination_stride);
    write(0x2c, canonical.geometry.resized_height << 16 | canonical.geometry.resized_width);
    write(0x30, canonical.geometry.top << 16 | canonical.geometry.left);
    issue(kStart);
    return pending_tag_;
}

void PreprocessController::validate_response(const ControlCompletion& result, std::uint32_t opcode) {
    if (result.tag != pending_tag_ || opcode != pending_opcode_ || result.status > 5 ||
        !(result.flags & 1U) || (result.flags & ~7U) != 0 || (result.stage_status & ~0x77fffU) != 0 ||
        ((result.stage_status >> 16) & 7U) > 5) {
        quarantine("preprocess response identity, status or drain proof is invalid");
    }
    for (unsigned stage = 0; stage < 5; ++stage) {
        if (((result.stage_status >> (stage * 3)) & 7U) > 5) {
            quarantine("preprocess response contains an invalid child status");
        }
    }
    if (result.status == 0 && (result.stage_status != 0 || (result.flags & 2U))) {
        quarantine("preprocess success contradicts a stage or configuration error");
    }
    if (opcode == kStart) {
        if (result.source_bytes > expected_source_bytes_ ||
            result.destination_bytes > expected_destination_bytes_ ||
            (result.status == 0 && (result.source_bytes != expected_source_bytes_ ||
                                   result.destination_bytes != expected_destination_bytes_))) {
            quarantine("preprocess byte counts contradict its command");
        }
    } else if (result.source_bytes != 0 || result.destination_bytes != 0 || result.stage_status != 0) {
        quarantine("map command unexpectedly reports DMA activity");
    }
}

std::optional<ControlCompletion> PreprocessController::poll() {
    if (phase_ != ControlPhase::Configuring && phase_ != ControlPhase::Running) {
        throw std::logic_error("no pollable preprocessing command");
    }
    const auto state = read(kState);
    if (read(kErrors) != 0 || (state & ~0x1fU) != 0) {
        quarantine("preprocess CSR reports an error");
    }
    if (!(state & 1U) || bool(state & 0x10U) != stop_requested_) {
        quarantine("preprocess command ownership or STOP changed unexpectedly");
    }
    if (!(state & 8U)) {
        return std::nullopt;
    }
    ControlCompletion result{};
    result.kind = phase_ == ControlPhase::Configuring ? CompletionKind::Configuration : CompletionKind::Frame;
    result.status = read(kResponse);
    result.tag = read(kResponse + 4);
    result.source_bytes = read(kResponse + 8);
    result.destination_bytes = read(kResponse + 12);
    result.stage_status = read(kResponse + 16);
    result.flags = read(kResponse + 20);
    const auto opcode = read(kResponse + 24);
    validate_response(result, opcode);
    const std::uint32_t expected_count = completed_count_ + 1U;
    if (read(kIssued) != expected_count || read(kCompleted) != expected_count) {
        quarantine("preprocess transaction counters changed outside their owner");
    }
    write(kAck, result.tag);
    completed_count_ = expected_count;
    check_hardware_idle();
    if (phase_ == ControlPhase::Configuring) {
        if (result.status == 0 && stop_requested_) {
            result.status = kStopped;
        }
        if (result.status == 0) {
            const auto map_count = configuration_->horizontal_map.size() + configuration_->vertical_map.size();
            if (configuration_step_ < map_count) {
                ++configuration_step_;
                issue_map_entry();
                return std::nullopt;
            }
            configured_geometry_ = configuration_->geometry;
        }
        configuration_.reset();
    }
    phase_ = ControlPhase::Idle;
    return result;
}

void PreprocessController::request_stop() {
    if (phase_ == ControlPhase::Uninitialized) {
        throw std::logic_error("uninitialized preprocess controller");
    }
    stop_requested_ = true;
    write(kStop, 1);
}

void PreprocessController::clear_stop() {
    require_idle();
    check_hardware_idle();
    write(kStop, 0);
    if (read(kStop) != 0) {
        quarantine("preprocess STOP release did not complete");
    }
    stop_requested_ = false;
}

} // namespace yolo26
