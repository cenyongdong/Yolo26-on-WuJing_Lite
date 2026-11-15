#include "yolo26/capture.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace yolo26 {
namespace {

int control(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ::ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

void checked_control(int fd, unsigned long request, void* argument, const char* operation) {
    if (control(fd, request, argument) < 0) {
        throw std::system_error(errno, std::generic_category(), operation);
    }
}

v4l2_buffer empty_buffer(std::uint32_t index = 0) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    return buffer;
}

} // namespace

std::uint64_t monotonic_ns() {
    timespec time{};
    if (::clock_gettime(CLOCK_MONOTONIC, &time) < 0) {
        throw std::system_error(errno, std::generic_category(), "clock_gettime");
    }
    return std::uint64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
}

struct CameraCapture::Impl {
    struct Mapping {
        void* address = MAP_FAILED;
        std::size_t bytes = 0;
    };
    struct Completed {
        v4l2_buffer buffer;
        std::uint64_t sequence;
        std::uint64_t captured_ns;
        bool driver_clock;
    };
    int fd = -1;
    bool streaming = false;
    bool cleanup_ok = true;
    bool have_sequence = false;
    std::uint32_t last_driver_sequence = 0;
    std::uint64_t sequence = 0, epoch = 0;
    CaptureFormat format{};
    CaptureCounters counters;
    std::vector<Mapping> mappings;

    bool shutdown() noexcept {
        if (streaming) {
            auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (control(fd, VIDIOC_STREAMOFF, &type) < 0) {
                cleanup_ok = false;
            }
            streaming = false;
        }
        // Linux V4L2 close 解除驱动缓冲锁定；先关闭设备，再撤销用户映射。
        // 这仅适用于内核 V4L2 所有的缓冲，不推广到自研 PL/SDK 分配的生命周期。
        if (fd >= 0) {
            if (::close(fd) < 0) {
                cleanup_ok = false;
            }
            fd = -1;
        }
        for (auto& mapping : mappings) {
            if (mapping.address != MAP_FAILED) {
                if (::munmap(mapping.address, mapping.bytes) < 0) {
                    cleanup_ok = false;
                }
                mapping.address = MAP_FAILED;
            }
        }
        return cleanup_ok;
    }

    ~Impl() {
        shutdown();
    }

    void queue(std::uint32_t index) {
        auto buffer = empty_buffer(index);
        checked_control(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
    }

    std::optional<Completed> inspect(v4l2_buffer buffer) {
        if (buffer.index >= mappings.size()) {
            throw std::runtime_error("driver returned invalid buffer index");
        }
        ++counters.dequeued;
        const auto difference = std::uint32_t(buffer.sequence - last_driver_sequence);
        if (have_sequence && (difference == 0 || difference >= 0x80000000U)) {
            ++counters.duplicate_or_reordered;
            return std::nullopt;
        }
        if (have_sequence) {
            counters.driver_sequence_gaps += difference - 1;
        }
        have_sequence = true;
        last_driver_sequence = buffer.sequence;
        ++sequence;
        if (buffer.flags & V4L2_BUF_FLAG_ERROR) {
            ++counters.flagged_error;
            return std::nullopt;
        }
        const auto extent = std::uint64_t(format.stride) * (format.height - 1) + format.width * 2;
        if (buffer.bytesused < extent || buffer.bytesused > mappings[buffer.index].bytes ||
            buffer.bytesused > format.size_image) {
            ++counters.invalid_bytes;
            return std::nullopt;
        }
        const auto observed = monotonic_ns();
        auto timestamp = observed;
        const bool driver_clock = (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
        if (driver_clock) {
            if (buffer.timestamp.tv_sec < 0 || std::uint64_t(buffer.timestamp.tv_sec) > observed / 1000000000 ||
                buffer.timestamp.tv_usec < 0 || buffer.timestamp.tv_usec >= 1000000) {
                ++counters.invalid_timestamp;
                return std::nullopt;
            }
            timestamp = std::uint64_t(buffer.timestamp.tv_sec) * 1000000000 + buffer.timestamp.tv_usec * 1000;
            if (timestamp == 0 || timestamp > observed) {
                ++counters.invalid_timestamp;
                return std::nullopt;
            }
        }
        return Completed{buffer, sequence, timestamp, driver_clock};
    }
};

CameraCapture::CameraCapture(const CaptureConfig& config) : impl_(std::make_unique<Impl>()) {
    if (config.epoch == 0 || config.width == 0 || config.width > 1280 || (config.width & 1) ||
        config.height == 0 || config.height > 720 || config.fps == 0 || config.fps > 60) {
        throw std::invalid_argument("invalid UVC capture configuration");
    }
    auto& state = *impl_;
    state.epoch = config.epoch;
    state.fd = ::open(config.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (state.fd < 0) {
        throw std::system_error(errno, std::generic_category(), "open UVC camera");
    }
    v4l2_capability capability{};
    checked_control(state.fd, VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
    const auto caps = capability.capabilities & V4L2_CAP_DEVICE_CAPS ? capability.device_caps : capability.capabilities;
    if (std::strncmp(reinterpret_cast<const char*>(capability.driver), "uvcvideo", sizeof(capability.driver)) != 0 ||
        !(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING)) {
        throw std::runtime_error("selected device is not a streaming UVC capture node");
    }
    if (!config.expected_bus_info.empty() && (config.expected_bus_info.size() >= sizeof(capability.bus_info) ||
        std::strncmp(reinterpret_cast<const char*>(capability.bus_info), config.expected_bus_info.c_str(),
                     sizeof(capability.bus_info)) != 0)) {
        throw std::runtime_error("UVC port identity changed after preflight");
    }
    v4l2_format requested{};
    requested.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    requested.fmt.pix.width = config.width;
    requested.fmt.pix.height = config.height;
    requested.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    requested.fmt.pix.field = V4L2_FIELD_NONE;
    const bool extended_format = (caps & V4L2_CAP_EXT_PIX_FORMAT) != 0;
    if (extended_format) {
        requested.fmt.pix.priv = V4L2_PIX_FMT_PRIV_MAGIC;
    }
    checked_control(state.fd, VIDIOC_S_FMT, &requested, "VIDIOC_S_FMT");
    const auto& pixels = requested.fmt.pix;
    const auto extent = std::uint64_t(pixels.bytesperline) * (pixels.height ? pixels.height - 1 : 0) + pixels.width * 2;
    if (pixels.pixelformat != V4L2_PIX_FMT_YUYV || pixels.field != V4L2_FIELD_NONE ||
        pixels.width == 0 || pixels.width > 1280 || (pixels.width & 1) || pixels.height == 0 || pixels.height > 720 ||
        pixels.bytesperline < pixels.width * 2 || pixels.sizeimage < extent || pixels.sizeimage > 8 * 1024 * 1024) {
        throw std::runtime_error("negotiated UVC format exceeds supported YUYV geometry");
    }
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    checked_control(state.fd, VIDIOC_G_PARM, &parameters, "VIDIOC_G_PARM");
    if (parameters.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) {
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator = config.fps;
        checked_control(state.fd, VIDIOC_S_PARM, &parameters, "VIDIOC_S_PARM");
    }
    const bool extended_valid = extended_format && pixels.priv == V4L2_PIX_FMT_PRIV_MAGIC;
    state.format = {pixels.width, pixels.height, pixels.bytesperline, pixels.sizeimage, 0,
                    parameters.parm.capture.timeperframe.numerator, parameters.parm.capture.timeperframe.denominator,
                    pixels.colorspace, extended_valid ? pixels.ycbcr_enc : std::uint32_t(V4L2_YCBCR_ENC_DEFAULT),
                    extended_valid ? pixels.quantization : std::uint32_t(V4L2_QUANTIZATION_DEFAULT),
                    extended_valid ? pixels.xfer_func : std::uint32_t(V4L2_XFER_FUNC_DEFAULT), extended_valid};
    v4l2_requestbuffers request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    request.count = 4;
    checked_control(state.fd, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
    if (request.count < 2 || request.count > 16) {
        throw std::runtime_error("driver returned unsupported capture buffer count");
    }
    state.mappings.resize(request.count);
    state.format.buffers = request.count;
    for (std::uint32_t index = 0; index < request.count; ++index) {
        auto buffer = empty_buffer(index);
        checked_control(state.fd, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
        if (buffer.length < pixels.sizeimage || buffer.length > 8 * 1024 * 1024) {
            throw std::runtime_error("driver returned invalid capture mapping length");
        }
        auto& mapping = state.mappings[index];
        mapping.bytes = buffer.length;
        mapping.address = ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, state.fd, buffer.m.offset);
        if (mapping.address == MAP_FAILED) {
            throw std::system_error(errno, std::generic_category(), "mmap capture buffer");
        }
        state.queue(index);
    }
}

CameraCapture::~CameraCapture() = default;
const CaptureFormat& CameraCapture::format() const { return impl_->format; }
const CaptureCounters& CameraCapture::counters() const { return impl_->counters; }

void CameraCapture::start() {
    if (impl_->fd < 0 || impl_->streaming) {
        throw std::logic_error("capture session cannot be started twice");
    }
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    checked_control(impl_->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    impl_->streaming = true;
}

FrameRef CameraCapture::next(FramePool& pool, int timeout_ms) {
    auto& state = *impl_;
    if (!state.streaming || timeout_ms < 0 || timeout_ms > 1000) {
        throw std::invalid_argument("capture is not running or timeout is invalid");
    }
    try {
        pollfd descriptor{state.fd, POLLIN, 0};
        const int result = ::poll(&descriptor, 1, timeout_ms);
        if (result < 0 && errno == EINTR) {
            return {};
        }
        if (result < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            throw std::runtime_error("UVC poll failed or device disconnected");
        }
        if (result == 0) {
            ++state.counters.poll_timeouts;
            return {};
        }
        std::optional<Impl::Completed> newest;
        // 有界地清空已完成队列；不等待未来帧，也不让高帧率生产者形成无限 drain 循环。
        for (std::size_t count = 0; count < state.mappings.size(); ++count) {
            auto buffer = empty_buffer();
            if (control(state.fd, VIDIOC_DQBUF, &buffer) < 0) {
                if (errno == EAGAIN) {
                    break;
                }
                throw std::system_error(errno, std::generic_category(), "VIDIOC_DQBUF");
            }
            auto complete = state.inspect(buffer);
            if (!complete) {
                state.queue(buffer.index);
                continue;
            }
            if (newest) {
                state.queue(newest->buffer.index);
                ++state.counters.latest_replaced;
            }
            newest = complete;
        }
        if (!newest) {
            return {};
        }
        auto writer = pool.try_acquire();
        if (!writer) {
            ++state.counters.pool_exhausted;
            state.queue(newest->buffer.index);
            return {};
        }
        if (writer->capacity() < newest->buffer.bytesused) {
            throw std::invalid_argument("Host frame pool is smaller than negotiated capture buffer");
        }
        std::memcpy(writer->data(), state.mappings[newest->buffer.index].address, newest->buffer.bytesused);
        const FrameInfo info{{state.epoch, newest->sequence}, newest->captured_ns, state.format.width,
                             state.format.height, state.format.stride, newest->buffer.bytesused, PixelFormat::Yuyv};
        auto frame = writer->publish(info);
        state.queue(newest->buffer.index);
        ++state.counters.published;
        if (newest->driver_clock) {
            ++state.counters.driver_timestamps;
        } else {
            ++state.counters.receive_timestamps;
        }
        return frame;
    } catch (...) {
        state.shutdown();
        throw;
    }
}

bool CameraCapture::stop() noexcept {
    return impl_->shutdown();
}

} // namespace yolo26
