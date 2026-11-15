#include "yolo26/frames.hpp"
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace yolo26 {
namespace detail {

struct FrameStorage {
    struct Slot {
        std::vector<std::uint8_t> bytes;
        bool owned = false;
    };
    std::mutex mutex;
    std::vector<Slot> slots;
    std::size_t in_use = 0, high_water = 0;
};

struct FrameOwner {
    FrameOwner(std::shared_ptr<FrameStorage> storage_value, std::size_t index_value)
        : storage(std::move(storage_value)), index(index_value) {}
    ~FrameOwner() {
        std::lock_guard<std::mutex> lock(storage->mutex);
        storage->slots[index].owned = false;
        --storage->in_use;
    }
    std::shared_ptr<FrameStorage> storage;
    std::size_t index;
};

} // namespace detail

FramePacket::FramePacket(std::shared_ptr<detail::FrameOwner> owner, FrameInfo info)
    : owner_(std::move(owner)), info_(info) {}

const std::uint8_t* FramePacket::data() const {
    return owner_->storage->slots[owner_->index].bytes.data();
}

FrameWriter::FrameWriter(std::shared_ptr<detail::FrameOwner> owner) : owner_(std::move(owner)) {}

std::uint8_t* FrameWriter::data() {
    if (!owner_) throw std::logic_error("writer already published or moved");
    return owner_->storage->slots[owner_->index].bytes.data();
}

std::size_t FrameWriter::capacity() const {
    if (!owner_) throw std::logic_error("writer already published or moved");
    return owner_->storage->slots[owner_->index].bytes.size();
}

FrameRef FrameWriter::publish(FrameInfo info) {
    if (!info.key.epoch || !info.captured_ns || !info.width || !info.height ||
        info.width > 1280 || info.height > 720 || !info.bytes_used || info.bytes_used > capacity()) {
        throw std::invalid_argument("invalid completed frame identity or extent");
    }
    if (info.format != PixelFormat::Mjpeg) {
        if (info.format != PixelFormat::Yuyv && info.format != PixelFormat::Rgb888 &&
            info.format != PixelFormat::Rgb565) {
            throw std::invalid_argument("unknown pixel format");
        }
        const std::size_t row_bytes = info.width * (info.format == PixelFormat::Rgb888 ? 3 : 2);
        if ((info.format == PixelFormat::Yuyv && info.width % 2) || info.stride < row_bytes ||
            std::uint64_t(info.stride) * (info.height - 1) + row_bytes > info.bytes_used) {
            throw std::invalid_argument("incomplete or malformed image rows");
        }
    }
    // 分配失败时 writer 仍保留原槽位，调用方可以丢弃或重试普通 Host 发布。
    FrameRef completed(new FramePacket(owner_, info));
    owner_.reset();
    return completed;
}

FramePool::FramePool(std::size_t slots, std::size_t bytes_per_slot)
    : storage_(std::make_shared<detail::FrameStorage>()) {
    if (!slots || !bytes_per_slot || slots > std::numeric_limits<std::size_t>::max() / bytes_per_slot) {
        throw std::invalid_argument("invalid frame pool capacity");
    }
    storage_->slots.resize(slots);
    for (auto& slot : storage_->slots) {
        slot.bytes.resize(bytes_per_slot);
    }
}

std::optional<FrameWriter> FramePool::try_acquire() {
    std::lock_guard<std::mutex> lock(storage_->mutex);
    for (std::size_t index = 0; index < storage_->slots.size(); ++index) {
        auto& slot = storage_->slots[index];
        if (!slot.owned) {
            // 先构造共享所有者，成功后再占槽，避免 bad_alloc 留下无法归还的槽位。
            auto owner = std::make_shared<detail::FrameOwner>(storage_, index);
            slot.owned = true;
            ++storage_->in_use;
            if (storage_->in_use > storage_->high_water) {
                storage_->high_water = storage_->in_use;
            }
            return FrameWriter(std::move(owner));
        }
    }
    return std::nullopt;
}

std::size_t FramePool::available() const {
    std::lock_guard<std::mutex> lock(storage_->mutex);
    return storage_->slots.size() - storage_->in_use;
}

std::size_t FramePool::high_water() const {
    std::lock_guard<std::mutex> lock(storage_->mutex);
    return storage_->high_water;
}

} // namespace yolo26
