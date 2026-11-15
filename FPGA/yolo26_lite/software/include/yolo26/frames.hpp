#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace yolo26 {
namespace detail {
struct FrameStorage;
struct FrameOwner;
}

enum class PixelFormat { Yuyv, Rgb888, Mjpeg, Rgb565 };

struct FrameKey {
    std::uint64_t epoch;
    std::uint64_t sequence;
    bool operator==(const FrameKey& other) const {
        return epoch == other.epoch && sequence == other.sequence;
    }
};

struct FrameInfo {
    FrameKey key;
    std::uint64_t captured_ns;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t stride;
    std::size_t bytes_used;
    PixelFormat format;
};

// 只读完整帧。复制共享引用不会复制图像，最后一个引用归还后槽位才能复用。
class FramePacket {
public:
    const FrameInfo& info() const { return info_; }
    const std::uint8_t* data() const;

private:
    friend class FrameWriter;
    FramePacket(std::shared_ptr<detail::FrameOwner> owner, FrameInfo info);
    std::shared_ptr<detail::FrameOwner> owner_;
    FrameInfo info_;
};
using FrameRef = std::shared_ptr<const FramePacket>;

// 唯一填充者；publish 后不可再写。放弃未完成填充时自动归还普通 Host 内存。
class FrameWriter {
public:
    FrameWriter(FrameWriter&&) noexcept = default;
    FrameWriter& operator=(FrameWriter&&) noexcept = default;
    FrameWriter(const FrameWriter&) = delete;
    FrameWriter& operator=(const FrameWriter&) = delete;
    std::uint8_t* data();
    std::size_t capacity() const;
    FrameRef publish(FrameInfo info);

private:
    friend class FramePool;
    explicit FrameWriter(std::shared_ptr<detail::FrameOwner> owner);
    std::shared_ptr<detail::FrameOwner> owner_;
};

// 仅管理普通 CPU 图像内存，不管理仍可能被硬件访问的 SDK／DMA 内存。
// 满池返回空，调用方丢帧；不等待消费者，不扩容。
class FramePool {
public:
    FramePool(std::size_t slots, std::size_t bytes_per_slot);
    std::optional<FrameWriter> try_acquire();
    std::size_t available() const;
    std::size_t high_water() const;

private:
    std::shared_ptr<detail::FrameStorage> storage_;
};

} // namespace yolo26
