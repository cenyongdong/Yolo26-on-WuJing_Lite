#include "yolo26/video_engine.hpp"
#include "yolo26/compose.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {

using namespace yolo26;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class Action> void rejects(Action action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "invalid video operation was accepted");
}

// 场景驱动的公开 CSR 模型：完成与采用由测试显式推进，不模拟像素算法或 SDK 内部实现。
class PreprocessRegisters : public RegisterAccess {
public:
    std::array<std::uint32_t, 35> words{};
    const ModelGeometry model;
    unsigned opcode = 0, commands = 0;
    explicit PreprocessRegisters(ModelGeometry geometry = kSquareModel) : model(geometry) {
        at(0) = 0x59323601;
        at(0x80) = 0x59324c01;
        at(0x84) = 0x30;
        at(0x88) = model.height << 16 | model.width;
    }
    std::uint32_t& at(unsigned offset) { return words.at(offset / 4); }
    std::uint32_t read32(std::uint32_t offset) override { return at(offset); }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        if (offset == 4) {
            require(!(at(8) & 15), "overlapping preprocess command");
            opcode = value;
            ++commands;
            ++at(0x60);
            at(8) = (at(8) & 16) | 5;
        } else if (offset == 0x5c) {
            require((at(8) & 8) && value == at(0x44), "invalid preprocess ACK");
            at(8) &= 16;
        } else if (offset == 0x3c) {
            at(offset) = value;
            at(8) = (at(8) & 15) | (value ? 16U : 0U);
        } else {
            at(offset) = value;
        }
    }
    void complete() {
        require((at(8) & 5) == 5, "no pending preprocessing operation");
        at(0x40) = 0;
        at(0x44) = at(0x10);
        at(0x48) = opcode == 1 ? (at(0x1c) >> 16) * (at(0x1c) & 65535) * 2 : 0;
        at(0x4c) = opcode == 1 ? static_cast<std::uint32_t>(model.rgb_elements()) : 0;
        at(0x54) = 1 | (at(0x3c) ? 4U : 0U);
        at(0x58) = opcode;
        at(8) = (at(8) & 16) | 9;
        ++at(0x64);
    }
};

class DisplayRegisters : public RegisterAccess {
public:
    std::array<std::uint32_t, 20> words{};
    DisplaySnapshot actual{0xc1, 0, 0, 0, 0};
    std::uint32_t next_address = 0, next_id = 0, stop_id = 0;
    DisplayRegisters() {
        words[0] = 0x48444d01;
        at(0x30) = 3;
        at(0x40) = 0x48445301;
        at(0x44) = 74250000;
        at(0x48) = 371250000;
        at(0x4c) = 720U << 16 | 1280U;
    }
    std::uint32_t& at(unsigned offset) { return words.at(offset / 4); }
    std::uint32_t read32(std::uint32_t offset) override { return at(offset); }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        at(offset) = value;
        if (offset == 0x0c) {
            require(at(0x24) == at(0x28), "overlapping HDMI command");
            ++at(0x24);
        }
    }
    void complete() {
        require(at(0x24) == at(0x28) + 1, "no pending HDMI command");
        at(0x10) = actual.status;
        at(0x14) = actual.stopped_id;
        at(0x18) = actual.accepted_address;
        at(0x1c) = actual.accepted_id;
        at(0x20) = actual.faults;
        if (at(0x0c) == 1 || at(0x0c) == 2) {
            next_address = at(4);
            next_id = at(8);
            actual.status = 0x42 | (actual.accepted_id ? 0x100 : 0);
        } else if (at(0x0c) == 3) {
            stop_id = at(8);
        }
        ++at(0x28);
    }
    void adopt() {
        actual.status = 0x142;
        actual.accepted_address = next_address;
        actual.accepted_id = next_id;
    }
    void drain() {
        require(stop_id != 0, "no hardware STOP to drain");
        actual.status = 0xc1;
        actual.stopped_id = stop_id;
    }
};

class Memory : public VideoMemoryAccess {
public:
    explicit Memory(ModelGeometry model = kSquareModel) : model_bytes(model.rgb_elements()) {}
    const std::size_t model_bytes;
    unsigned writes = 0, reads = 0, display_reads = 0;
    bool throw_read = false, throw_write = false;
    void write(std::size_t slot, const std::uint8_t* source, std::size_t bytes) override {
        require(slot == 0 || (slot >= 4 && slot <= 6), "unexpected memory write destination");
        require(source && bytes, "empty memory write");
        ++writes;
        if (throw_write) {
            throw std::runtime_error("injected write outcome unknown");
        }
    }
    void read(std::size_t slot, std::uint8_t* destination, std::size_t bytes) override {
        if (slot >= 4 && slot <= 6 && bytes == Compositor720p::kBytes) {
            ++display_reads;
            std::fill(destination, destination + bytes, 28);
            return;
        }
        require(slot >= 1 && slot <= 3 && bytes == model_bytes, "unexpected model read");
        ++reads;
        destination[0] = 114;
        if (throw_read) {
            throw std::runtime_error("injected read outcome unknown");
        }
        std::fill(destination, destination + bytes, 114);
    }
};

VideoAllocations allocations() {
    VideoAllocations result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = {0x31000000 + index * 0x400000, 0x400000};
    }
    return result;
}

FrameRef raw_frame(FramePool& pool, unsigned sequence) {
    auto writer = pool.try_acquire();
    require(bool(writer), "raw test pool exhausted");
    std::fill(writer->data(), writer->data() + 16, 128);
    return writer->publish({{26, sequence}, 100, 4, 2, 8, 16, PixelFormat::Yuyv});
}

FrameRef display_frame(FramePool& pool, unsigned sequence) {
    auto writer = pool.try_acquire();
    require(bool(writer), "display test pool exhausted");
    return writer->publish({{26, sequence}, 100, 1280, 720, 2560, Compositor720p::kBytes, PixelFormat::Rgb565});
}

PreparedFrame finish(VideoEngine& engine, PreprocessRegisters& registers, FramePool& pool) {
    for (unsigned count = 0; count < 1300; ++count) {
        require(!engine.poll_preprocess(pool), "preprocess finished before hardware response");
        registers.complete();
        auto result = engine.poll_preprocess(pool);
        if (result) {
            return *result;
        }
    }
    throw std::runtime_error("preprocess integration did not finish within its finite command count");
}

void drain_display(VideoEngine& engine, DisplayRegisters& display) {
    engine.request_stop();
    display.complete(); // 先收回初始化快照，再发 STOP。
    engine.pump_display();
    display.complete(); // STOP 应答仍是命令执行前的快照。
    engine.pump_display();
    require(engine.must_retain_resources(), "error cleanup skipped the display drain proof");
    display.drain();
    display.complete();
    engine.pump_display();
    require(!engine.must_retain_resources(), "proven preprocessing failure could not finish display STOP");
}

void completed_failure_cleanup(bool device_error) {
    PreprocessRegisters preprocess;
    DisplayRegisters display;
    Memory memory;
    FramePool raw(1, 16), model(1, device_error ? kModelElements : 8);
    VideoEngine engine(preprocess, display, memory, allocations());
    engine.initialize();
    auto frame = raw_frame(raw, 1);
    std::weak_ptr<const FramePacket> borrowed = frame;
    engine.begin_preprocess(std::move(frame), ColorMatrix::Bt601Limited);
    if (device_error) {
        unsigned commands = 0;
        while (preprocess.opcode != 1) {
            require(++commands <= 1300, "test did not reach the frame command");
            preprocess.complete();
            require(!engine.poll_preprocess(model), "map configuration unexpectedly published a frame");
        }
        preprocess.complete();
        preprocess.at(0x40) = 2; // DMA 响应错误，但控制器仍提供完整 tag/计数/排空证明。
        preprocess.at(0x50) = 2U | (1U << 16);
        rejects([&] { engine.poll_preprocess(model); });
    } else {
        rejects([&] { finish(engine, preprocess, model); }); // 硬件成功后才发现 Host 输入池太小。
    }
    require(!engine.preprocess_busy() && borrowed.expired() && raw.available() == 1 &&
            model.available() == 1 && memory.reads == 0,
            "completed failure retained a pollable task or started an invalid Host transfer");
    rejects([&] { engine.begin_preprocess(raw_frame(raw, 2), ColorMatrix::Bt601Limited); });
    drain_display(engine, display);
}

} // namespace

int main() {
    {
        constexpr auto shape = kRectangularModel;
        FramePool source(1, 16), model(1, shape.rgb_elements());
        PreprocessRegisters preprocess(shape);
        DisplayRegisters display;
        Memory memory(shape);
        auto ranges = allocations();
        for (std::size_t i = 1; i <= 3; ++i) {
            ranges[i].capacity = shape.rgb_elements();
        }
        VideoEngine engine(preprocess, display, memory, ranges, shape);
        preprocess.at(0x88) = 512U << 16 | 384U;
        rejects([&] { engine.initialize(); });
        require(preprocess.commands == 0 && display.at(0x24) == 0 && !engine.must_retain_resources(),
                "mismatched model and PL accessed the frame path");
        preprocess.at(0x88) = shape.height << 16 | shape.width;
        VideoEngine rect(preprocess, display, memory, ranges, shape);
        rect.initialize();
        rect.begin_preprocess(raw_frame(source, 1), ColorMatrix::Bt601Limited);
        const auto prepared = finish(rect, preprocess, model);
        const auto& info = prepared.rgb->info();
        require(info.width == 512 && info.height == 384 && info.stride == 1536 && info.bytes_used == 589824 &&
                info.key == prepared.original->info().key && preprocess.at(0x28) == 1536 &&
                prepared.geometry.top == 64 && memory.reads == 1,
                "rectangular command, result length or identity differs");
        drain_display(rect, display);
    }
    FramePool raw_pool(6, 16), model_pool(3, kModelElements), display_pool(3, Compositor720p::kBytes);
    PreprocessRegisters preprocess;
    DisplayRegisters display;
    Memory memory;
    VideoEngine engine(preprocess, display, memory, allocations());
    auto overlap = allocations();
    overlap[1] = overlap[0];
    rejects([&] { VideoEngine bad(preprocess, display, memory, overlap); });
    engine.initialize();
    auto frame = raw_frame(raw_pool, 1);
    std::weak_ptr<const FramePacket> original = frame;
    require(engine.begin_preprocess(frame, ColorMatrix::Bt601Limited), "raw frame not accepted");
    frame.reset();
    require(!original.expired() && engine.preprocess_busy() &&
            !engine.begin_preprocess({}, ColorMatrix::Bt601Limited), "raw ownership or busy policy failed");
    auto prepared = finish(engine, preprocess, model_pool);
    require(prepared.disposition == PreprocessDisposition::Ready && prepared.original->info().key == FrameKey{26, 1} &&
            prepared.rgb->info().key == prepared.original->info().key && prepared.rgb->data()[0] == 114 &&
            prepared.geometry.top == 160 && memory.reads == 1 && !engine.preprocess_busy(),
            "prepared frame identity, letterbox or publication differs");
    const auto first_commands = preprocess.commands;
    require(engine.begin_preprocess(raw_frame(raw_pool, 2), ColorMatrix::Bt709Full), "second frame rejected");
    auto second = finish(engine, preprocess, model_pool);
    require(preprocess.commands == first_commands + 1 && second.matrix == ColorMatrix::Bt709Full,
            "unchanged geometry redundantly reconfigured or lost its color matrix");
    rejects([&] { engine.begin_preprocess(raw_frame(raw_pool, 2), ColorMatrix::Bt601Limited); });

    // 填满三槽 Host 输入池后，完成帧有界丢弃；绝不扩容或阻塞继续等待消费者。
    auto held = model_pool.try_acquire();
    require(bool(held), "third model slot unavailable");
    require(engine.begin_preprocess(raw_frame(raw_pool, 3), ColorMatrix::Bt601Limited), "third frame rejected");
    auto dropped = finish(engine, preprocess, model_pool);
    require(dropped.disposition == PreprocessDisposition::NoHostBuffer && !dropped.rgb && memory.reads == 2,
            "exhausted Host model pool did not drop without a transfer");

    require(engine.upload_display(display_frame(display_pool, 1)), "first composed frame rejected");
    require(engine.upload_display(display_frame(display_pool, 2)) && engine.replaced_display_frames() == 1,
            "latest complete display frame did not replace its pending predecessor");
    display.complete(); // Initial snapshot; pump now submits START with the latest frame.
    require(!engine.pump_display(), "initial snapshot counted as an adopted frame");
    display.complete(); // START reply contains its pre-command idle snapshot.
    require(!engine.pump_display(), "START reply counted as frame adoption");
    display.complete();
    require(!engine.pump_display(), "pre-boundary snapshot counted as frame adoption");
    display.adopt();
    display.complete();
    const auto adopted = engine.pump_display();
    require(adopted && *adopted == FrameKey{26, 2}, "latest frame or same-frame identity was lost");
    require(engine.last_adoption() && engine.last_adoption()->frame.key == *adopted &&
            engine.last_adoption()->frame.captured_ns == 100 && engine.last_adoption()->upload_started_ns >= 100,
            "adoption timing was taken from a replaced or unrelated display frame");
    rejects([&] { engine.read_adopted_pixels({26, 1}); });
    const auto adopted_pixels = engine.read_adopted_pixels(*adopted);
    require(adopted_pixels.size() == Compositor720p::kBytes && adopted_pixels.back() == 28 &&
            memory.display_reads == 1 && engine.last_adoption()->snapshot.accepted_id == display.actual.accepted_id,
            "adopted display read does not match the scan buffer identity");

    engine.request_stop();
    display.complete(); // Already pending snapshot; STOP is issued only after it returns.
    engine.pump_display();
    display.complete(); // Pre-STOP snapshot cannot free scanning buffers.
    engine.pump_display();
    require(engine.must_retain_resources(), "STOP command reply released active scan memory");
    display.drain();
    display.complete();
    engine.pump_display();
    require(!engine.must_retain_resources(), "tagged drained snapshot did not release the owned transport");
    require(model_pool.high_water() == 3 && raw_pool.high_water() >= 3 && display_pool.high_water() >= 1,
            "pool high-water missed ownership before a later sampling point");
    rejects([&] { engine.upload_display(display_frame(display_pool, 4)); });

    // 部分 SDK 读异常后保留 Host 写入目标和原图；STOP 不能被当作该读事务完成证明。
    PreprocessRegisters failed_preprocess;
    DisplayRegisters failed_display;
    Memory failed_memory;
    FramePool failed_models(3, kModelElements);
    VideoEngine failed(failed_preprocess, failed_display, failed_memory, allocations());
    failed.initialize();
    failed.begin_preprocess(raw_frame(raw_pool, 10), ColorMatrix::Bt601Limited);
    failed_memory.throw_read = true;
    rejects([&] { finish(failed, failed_preprocess, failed_models); });
    require(failed.must_retain_resources() && failed_models.available() == 2,
            "unknown native read released its Host target or DMA resources");
    rejects([&] { failed.request_stop(); });

    PreprocessRegisters cancelled_preprocess;
    DisplayRegisters cancelled_display;
    Memory cancelled_memory;
    VideoEngine cancelled(cancelled_preprocess, cancelled_display, cancelled_memory, allocations());
    cancelled.initialize();
    cancelled.begin_preprocess(raw_frame(raw_pool, 11), ColorMatrix::Bt601Limited);
    cancelled.request_stop();
    cancelled_preprocess.complete();
    const auto stopped = cancelled.poll_preprocess(model_pool);
    require(stopped && stopped->disposition == PreprocessDisposition::Stopped && !stopped->rgb &&
            cancelled_memory.reads == 0 && !cancelled.preprocess_busy(), "STOP during configuration published data");
    cancelled_display.complete();
    cancelled.pump_display();
    cancelled_display.complete();
    cancelled.pump_display();
    cancelled_display.drain();
    cancelled_display.complete();
    cancelled.pump_display();
    require(!cancelled.must_retain_resources(), "configuration STOP did not close independently of model pool");
    completed_failure_cleanup(true);
    completed_failure_cleanup(false);
    std::cout << "video engine identity, bounded pools, mailbox adoption and retention passed\n";
}
