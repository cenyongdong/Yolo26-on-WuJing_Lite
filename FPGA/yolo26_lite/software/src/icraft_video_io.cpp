#include "icraft_video_io.hpp"
#include "yolo26/compose.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo26::detail {
namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

bool overlaps(const DmaRange& left, const DmaRange& right) {
    // 用起点差计算，避免由 SDK 返回的不完整范围在加法时回绕。
    if (!left.capacity || !right.capacity) {
        return false;
    }
    return left.address <= right.address ? right.address - left.address < left.capacity :
                                           left.address - right.address < right.capacity;
}

} // namespace

IcraftVideoIo::IcraftVideoIo(const icraft::xrt::Device& device, pose180_pipeline::ReleaseGuard& activity)
    : device_(device), activity_(activity) {}

void IcraftVideoIo::require_owner_and_idle_transport() const {
    require(owner_ == std::this_thread::get_id(), "video SDK access outside its owning thread");
    require(activity_.permitted(), "previous SDK operation is unproved; retain video transport");
}

void IcraftVideoIo::initialize(ModelGeometry model) {
    require_owner_and_idle_transport();
    require(!initialization_attempted_, "UDMA allocation initialization is single-use");
    initialization_attempted_ = true;
    activity_.begin("udma_region_acquire");
    region_ = device_.getMemRegion("udma");
    activity_.complete();
    require(std::string(region_->typeKey()) == "AXIZG330AIUdmaMemRegionNode", "PS UDMA memory region differs");
    // 公共管理器可同时持有模型／设备的分配。只复制实际 UDMA 区的地址范围，不保留其它 MemChunk。
    std::vector<DmaRange> occupied;
    for (const auto& chunk : region_.memManager().getAllMemChunk()) {
        if (chunk.isOn(region_)) {
            occupied.push_back({chunk->begin.addr(), chunk->byte_size});
        }
    }
    for (std::size_t index = 0; index < chunks_.size(); ++index) {
        const std::uint64_t bytes = index == 0 ? 4 * 1024 * 1024 :
                                    index < 4 ? model.rgb_elements() : Compositor720p::kBytes;
        activity_.begin("udma_allocate");
        chunks_[index] = region_.malloc(bytes, false, 64); // 不自动 free；只有联合完成证明后显式释放。
        allocated_[index] = true;
        activity_.complete();
        const auto address = chunks_[index]->begin.addr();
        const DmaRange allocated{address, chunks_[index]->byte_size};
        allocations_[index] = allocated;
        require(chunks_[index].isOn(region_) && chunks_[index]->byte_size == bytes &&
                address >= 0x30100000 && address <= 0x38100000 - bytes && (address & 63U) == 0,
                "PS UDMA allocation lies outside the qualified physical region");
        require(std::none_of(occupied.begin(), occupied.end(), [&](const auto& range) { return overlaps(range, allocated); }),
                "Video UDMA allocation overlaps an existing reservation");
        occupied.push_back(allocated);
    }
    ready_ = true;
}

void IcraftVideoIo::check_transfer(std::size_t slot, const void* pointer, std::size_t bytes) const {
    require_owner_and_idle_transport();
    require(ready_ && slot < chunks_.size() && allocated_[slot] && pointer && bytes > 0 &&
            bytes <= allocations_[slot].capacity, "video transfer exceeds its owned memory block");
}

void IcraftVideoIo::write(std::size_t slot, const std::uint8_t* source, std::size_t bytes) {
    check_transfer(slot, source, bytes);
    require(slot == 0 || slot >= 4, "cannot upload to a PL-owned model output");
    activity_.begin("udma_host_upload");
    chunks_[slot].write(0, reinterpret_cast<char*>(const_cast<std::uint8_t*>(source)), bytes);
    activity_.complete();
}

void IcraftVideoIo::read(std::size_t slot, std::uint8_t* destination, std::size_t bytes) {
    check_transfer(slot, destination, bytes);
    require(slot >= 1 && (slot <= 3 || bytes == Compositor720p::kBytes),
            "only complete model or display output can be read");
    activity_.begin(slot <= 3 ? "udma_model_readback" : "udma_adopted_display_readback");
    chunks_[slot].read(reinterpret_cast<char*>(destination), 0, bytes);
    activity_.complete();
}

bool IcraftVideoIo::release_known_idle() {
    require_owner_and_idle_transport();
    ready_ = false;
    for (std::size_t index = 0; index < chunks_.size(); ++index) {
        if (allocated_[index]) {
            // free 抛出时 guard 不清除，下次无法再次执行同一 free。
            activity_.begin("udma_release");
            chunks_[index].free();
            allocated_[index] = false;
            activity_.complete();
        }
    }
    if (std::any_of(allocations_.begin(), allocations_.end(), [](const auto& range) { return range.capacity != 0; })) {
        for (const auto& chunk : region_.memManager().getAllMemChunk()) {
            if (!chunk.isOn(region_)) {
                continue;
            }
            const DmaRange remaining{chunk->begin.addr(), chunk->byte_size};
            if (std::any_of(allocations_.begin(), allocations_.end(),
                            [&](const auto& range) { return overlaps(range, remaining); })) {
                activity_.begin("owned_video_udma_remains_after_release");
                return false;
            }
        }
    }
    return true;
}

std::uint32_t IcraftVideoIo::Registers::read32(std::uint32_t offset) {
    owner_.require_owner_and_idle_transport();
    require((offset & 3U) == 0 && offset < extent_, "MMIO read offset is unaligned or outside its peripheral");
    owner_.activity_.begin("video_register_read");
    const auto value = owner_.device_.defaultRegRegion().read(base_ + offset);
    owner_.activity_.complete();
    return static_cast<std::uint32_t>(value);
}

void IcraftVideoIo::Registers::write32(std::uint32_t offset, std::uint32_t value) {
    owner_.require_owner_and_idle_transport();
    require((offset & 3U) == 0 && offset < extent_, "MMIO write offset is unaligned or outside its peripheral");
    owner_.activity_.begin("video_register_write");
    owner_.device_.defaultRegRegion().write(base_ + offset, value);
    owner_.activity_.complete();
}

} // namespace yolo26::detail
