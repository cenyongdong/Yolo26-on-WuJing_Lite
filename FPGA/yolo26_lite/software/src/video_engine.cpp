#include "yolo26/video_engine.hpp"
#include "yolo26/compose.hpp"
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace yolo26 {
namespace {

VideoAllocations checked_allocations(VideoAllocations memory, ModelGeometry model) {
    if (!model.valid()) {
        throw std::invalid_argument("invalid video model geometry");
    }
    for (std::size_t index = 0; index < memory.size(); ++index) {
        const auto minimum = index == 0 ? 8U : index < 4 ? model.rgb_elements() : Compositor720p::kBytes;
        const auto& range = memory[index];
        if ((range.address & 63U) || range.address > std::numeric_limits<std::uint32_t>::max() ||
            range.capacity < minimum || range.capacity > (1ULL << 32) - range.address) {
            throw std::invalid_argument("video allocation is unaligned, incomplete or outside AXI address width");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (range.address < memory[prior].address + memory[prior].capacity &&
                memory[prior].address < range.address + range.capacity) {
                throw std::invalid_argument("video physical allocations overlap");
            }
        }
    }
    return memory;
}

bool same_source(const Letterbox& left, const Letterbox& right) {
    // 目标尺寸在本实例构造时固定；缩放表只随源尺寸改变。
    return left.source_width == right.source_width && left.source_height == right.source_height;
}

} // namespace

VideoEngine::VideoEngine(RegisterAccess& preprocess_registers, RegisterAccess& display_registers,
                         VideoMemoryAccess& memory, VideoAllocations allocations, ModelGeometry model)
    : preprocess_registers_(preprocess_registers), memory_(memory), model_(model),
      allocations_(checked_allocations(allocations, model)), preprocess_(preprocess_registers), display_(display_registers),
      display_buffers_({allocations_[4], allocations_[5], allocations_[6]}) {}

void VideoEngine::assert_owner() const {
    if (owner_ != std::this_thread::get_id()) {
        throw std::logic_error("video transport accessed outside its single owner");
    }
}

void VideoEngine::require_new_work() const {
    assert_owner();
    if (!initialized_ || stopping_ || failed_ || transfer_uncertain_) {
        throw std::logic_error("video engine cannot accept new work");
    }
}

void VideoEngine::initialize() {
    assert_owner();
    if (initialized_ || failed_) {
        throw std::logic_error("video initialization is single-use");
    }
    try {
        if (preprocess_registers_.read32(0x80) != 0x59324c01 || preprocess_registers_.read32(0x84) != 0x30) {
            throw std::runtime_error("YOLO SoC ABI or initial shared bus state differs");
        }
        if (preprocess_registers_.read32(0x88) != ((model_.height << 16) | model_.width)) {
            throw std::runtime_error("PL RGB geometry does not match the selected model");
        }
        preprocess_.initialize();
        display_.initialize();
        initialized_ = true;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

bool VideoEngine::begin_preprocess(FrameRef frame, ColorMatrix matrix) {
    require_new_work();
    if (source_) {
        return false;
    }
    if (!frame || frame->info().bytes_used > allocations_[0].capacity) {
        throw std::invalid_argument("raw frame does not fit its dedicated physical staging buffer");
    }
    const auto& info = frame->info();
    if (last_source_.epoch && (info.key.epoch != last_source_.epoch || info.key.sequence <= last_source_.sequence)) {
        throw std::invalid_argument("preprocessing frame is duplicated, reordered or from another run");
    }
    auto plan = make_preprocess_plan(allocations_[0], info.stride, info.width, info.height,
                                    info.format, matrix, allocations_[input_slot_],
                                    static_cast<std::uint32_t>(model_.rgb_row_bytes()), model_);
    source_ = std::move(frame);
    plan_ = std::move(plan);
    last_source_ = info.key;
    try {
        transfer_uncertain_ = true;
        memory_.write(0, source_->data(), info.bytes_used);
        transfer_uncertain_ = false;
        if (!configured_ || !same_source(*configured_, plan_->geometry)) {
            preprocess_.configure(*plan_);
        } else {
            preprocess_.start(*plan_);
        }
    } catch (...) {
        failed_ = true;
        throw;
    }
    return true;
}

PreparedFrame VideoEngine::finish_preprocess(FrameRef rgb, PreprocessDisposition disposition) {
    PreparedFrame result{std::move(source_), std::move(rgb), plan_->geometry, plan_->matrix, disposition};
    plan_.reset();
    reading_model_.reset();
    input_slot_ = input_slot_ == 3 ? 1 : input_slot_ + 1;
    return result;
}

std::optional<PreparedFrame> VideoEngine::poll_preprocess(FramePool& model_pool) {
    assert_owner();
    if (transfer_uncertain_) {
        throw std::logic_error("SDK transfer state is unknown; retain all video objects");
    }
    if (!source_) {
        return std::nullopt;
    }
    try {
        const auto completion = preprocess_.poll();
        if (!completion) {
            return std::nullopt;
        }
        if (completion->status != 0 && completion->status != 5) {
            throw std::runtime_error("preprocess completed with a hardware error; no image published");
        }
        if (stopping_ || completion->status == 5) {
            return finish_preprocess({}, PreprocessDisposition::Stopped);
        }
        if (completion->kind == CompletionKind::Configuration) {
            configured_ = plan_->geometry;
            preprocess_.start(*plan_);
            return std::nullopt;
        }
        reading_model_ = model_pool.try_acquire();
        if (!reading_model_) {
            return finish_preprocess({}, PreprocessDisposition::NoHostBuffer);
        }
        const auto bytes = model_.rgb_elements();
        if (reading_model_->capacity() < bytes) {
            throw std::invalid_argument("model Host pool does not hold a complete RGB888 input");
        }
        transfer_uncertain_ = true;
        memory_.read(input_slot_, reading_model_->data(), bytes);
        transfer_uncertain_ = false;
        const auto& source_info = source_->info();
        auto rgb = reading_model_->publish({source_info.key, source_info.captured_ns, model_.width, model_.height,
                                           static_cast<std::uint32_t>(model_.rgb_row_bytes()), bytes,
                                           PixelFormat::Rgb888});
        return finish_preprocess(std::move(rgb), PreprocessDisposition::Ready);
    } catch (...) {
        failed_ = true;
        // 控制器已经确认排空/ACK 后的错误仍须保留原异常，但不应继续 poll 已回到 idle 的任务。
        // 若 SDK 读取结果未知则必须保留 Host 目标；只有两条独立完成边界都已知才归还普通帧引用。
        if (!transfer_uncertain_ && !preprocess_.must_retain_resources()) {
            source_.reset();
            plan_.reset();
            reading_model_.reset();
        }
        throw;
    }
}

bool VideoEngine::upload_display(FrameRef rgb565) {
    require_new_work();
    if (!rgb565 || rgb565->info().format != PixelFormat::Rgb565 || rgb565->info().width != 1280 ||
        rgb565->info().height != 720 || rgb565->info().stride != 2560 ||
        rgb565->info().bytes_used != Compositor720p::kBytes) {
        throw std::invalid_argument("display upload must be one complete 720p RGB565 frame");
    }
    const auto upload = display_buffers_.begin_upload(rgb565->info().key);
    if (!upload) {
        return false;
    }
    uploading_display_ = std::move(rgb565);
    display_timing_[upload->slot] = {uploading_display_->info(), static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()), upload->slot, {}};
    try {
        transfer_uncertain_ = true;
        memory_.write(4 + upload->slot, uploading_display_->data(), Compositor720p::kBytes);
        transfer_uncertain_ = false;
        display_buffers_.finish_upload(*upload);
        uploading_display_.reset();
    } catch (...) {
        failed_ = true;
        display_buffers_.quarantine();
        throw;
    }
    return true;
}

std::optional<FrameKey> VideoEngine::pump_display() {
    assert_owner();
    if (transfer_uncertain_ || !initialized_) {
        throw std::logic_error("display transport is unknown or uninitialized");
    }
    if (display_buffers_.stopped()) {
        return std::nullopt;
    }
    std::optional<FrameKey> adopted;
    try {
        if (const auto reply = display_.poll()) {
            if (!reply->accepted()) {
                throw std::runtime_error("display command was rejected");
            }
            if (reply->command == DisplayCommand::Snapshot) {
                if (stopping_) {
                    display_buffers_.observe_stop(reply->snapshot);
                } else {
                    adopted = display_buffers_.observe(reply->snapshot);
                    if (adopted) {
                        for (const auto& timing : display_timing_) {
                            if (timing.frame.key == *adopted) {
                                last_adoption_ = timing;
                                last_adoption_->snapshot = reply->snapshot;
                            }
                        }
                        if (!last_adoption_ || !(last_adoption_->frame.key == *adopted)) {
                            throw std::runtime_error("adopted display frame has no matching upload identity");
                        }
                    }
                }
            }
        }
        if (!display_.ready() || display_buffers_.stopped()) {
            return adopted;
        }
        if (stopping_) {
            if (!stop_sent_) {
                stop_sent_ = display_.request_stop(stop_tag_);
            } else {
                display_.request_snapshot();
            }
        } else {
            if (!submission_) {
                submission_ = display_buffers_.begin_submit();
            }
            if (submission_) {
                if (display_.submit(*submission_)) {
                    submission_.reset();
                }
            } else {
                display_.request_snapshot();
            }
        }
    } catch (...) {
        failed_ = true;
        throw;
    }
    return adopted;
}

std::vector<std::uint8_t> VideoEngine::read_adopted_pixels(FrameKey frame) {
    require_new_work();
    if (!last_adoption_ || !(last_adoption_->frame.key == frame)) {
        throw std::invalid_argument("no matching HDMI adoption to read back");
    }
    const auto slot = 4 + last_adoption_->slot;
    if (last_adoption_->snapshot.accepted_address != allocations_[slot].address ||
        last_adoption_->snapshot.accepted_id == 0) {
        throw std::logic_error("HDMI adoption does not identify the uploaded buffer");
    }
    // 只在观察采用的同一所有者迭代中调用，任何后续 upload 前完成。未知传输保留目标。
    reading_display_.resize(Compositor720p::kBytes);
    transfer_uncertain_ = true;
    memory_.read(slot, reading_display_.data(), reading_display_.size());
    transfer_uncertain_ = false;
    return std::exchange(reading_display_, {});
}

void VideoEngine::request_stop() {
    assert_owner();
    if (stopping_) {
        return;
    }
    if (!initialized_ || transfer_uncertain_) {
        throw std::logic_error("cannot substitute a STOP for an unknown SDK transfer");
    }
    stopping_ = true;
    submission_.reset(); // 尚未写入的提交由最终 STOP 一并解除账本占有，不能提前 FREE。
    stop_tag_ = display_buffers_.begin_stop();
    preprocess_.request_stop();
}

bool VideoEngine::must_retain_resources() const noexcept {
    return transfer_uncertain_ || preprocess_.must_retain_resources() || display_.must_retain_resources() ||
           display_buffers_.must_retain_resources();
}

} // namespace yolo26
