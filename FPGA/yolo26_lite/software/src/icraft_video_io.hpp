#pragma once
#include "yolo26/video_engine.hpp"
#include "release_guard.hpp"
#include <icraft-xrt/core/device.h>
#include <array>

namespace yolo26::detail {

// SDK 类型仅在适配实现内部可见，外层流水线只拿 VideoEngine 的明确接口。
class IcraftVideoIo final : public VideoMemoryAccess {
public:
    class Registers final : public RegisterAccess {
    public:
        Registers(IcraftVideoIo& owner, std::uint64_t base, std::uint32_t extent)
            : owner_(owner), base_(base), extent_(extent) {}
        std::uint32_t read32(std::uint32_t offset) override;
        void write32(std::uint32_t offset, std::uint32_t value) override;
    private:
        IcraftVideoIo& owner_;
        std::uint64_t base_;
        std::uint32_t extent_;
    };

    IcraftVideoIo(const icraft::xrt::Device& device, pose180_pipeline::ReleaseGuard& activity);
    void initialize(ModelGeometry model);
    void write(std::size_t slot, const std::uint8_t* source, std::size_t bytes) override;
    void read(std::size_t slot, std::uint8_t* destination, std::size_t bytes) override;
    bool release_known_idle();
    const VideoAllocations& allocations() const noexcept { return allocations_; }
    Registers preprocess_registers{*this, 0x400b5000, 0x100};
    Registers display_registers{*this, 0x400b0000, 0x80};

private:
    void require_owner_and_idle_transport() const;
    void check_transfer(std::size_t slot, const void* pointer, std::size_t bytes) const;
    icraft::xrt::Device device_;
    icraft::xrt::MemRegion region_;
    std::array<icraft::xrt::MemChunk, 7> chunks_;
    std::array<bool, 7> allocated_{};
    VideoAllocations allocations_{};
    pose180_pipeline::ReleaseGuard& activity_;
    const std::thread::id owner_ = std::this_thread::get_id();
    bool initialization_attempted_ = false, ready_ = false;
};

} // namespace yolo26::detail
