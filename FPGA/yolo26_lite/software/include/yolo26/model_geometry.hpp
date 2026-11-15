#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace yolo26 {

inline constexpr std::array<std::uint32_t, 3> kDetectionStrides{8, 16, 32};
inline constexpr std::size_t kClasses = 80;
inline constexpr std::size_t kRawChannels = 4 + kClasses;

struct ModelGeometry {
    std::uint32_t width = 640;
    std::uint32_t height = 640;

    constexpr bool valid() const noexcept {
        return width != 0 && width <= 640 && height != 0 && height <= 640 &&
               width % kDetectionStrides.back() == 0 && height % kDetectionStrides.back() == 0;
    }
    constexpr std::size_t rgb_row_bytes() const noexcept { return std::size_t{width} * 3; }
    constexpr std::size_t rgb_elements() const noexcept { return rgb_row_bytes() * height; }
    constexpr std::size_t anchors() const noexcept {
        std::size_t count = 0;
        for (const auto stride : kDetectionStrides) {
            count += std::size_t{width / stride} * (height / stride);
        }
        return count;
    }
    constexpr std::size_t raw_elements() const noexcept { return kRawChannels * anchors(); }
    constexpr bool operator==(ModelGeometry other) const noexcept {
        return width == other.width && height == other.height;
    }
};

inline constexpr ModelGeometry kSquareModel{};
inline constexpr ModelGeometry kRectangularModel{512, 384};

constexpr bool is_square_model(ModelGeometry shape) noexcept {
    return shape == kSquareModel;
}

} // namespace yolo26
