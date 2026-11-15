#include "yolo26/display_control.hpp"
#include "yolo26/compose.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class Action>
void rejects(Action action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "invalid display control operation was accepted");
}

// 只模拟公开 CSR 行为：应答可独立延迟，快照内容由测试场景给定，命令写后也可抛异常。
class Registers : public yolo26::RegisterAccess {
public:
    Registers() {
        words[0] = 0x48444d01;
        words[0x30 / 4] = 3;
        words[0x34 / 4] = 1;
        words[0x40 / 4] = 0x48445301;
        words[0x44 / 4] = 74250000;
        words[0x48 / 4] = 371250000;
        words[0x4c / 4] = (720U << 16) | 1280U;
    }
    std::uint32_t read32(std::uint32_t offset) override {
        if (offset == 0x1c && tear_snapshot) {
            ++words[0x28 / 4];
        }
        return words.at(offset / 4);
    }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        writes.push_back(offset);
        words.at(offset / 4) = value;
        if (offset == 0x0c) {
            ++words[0x24 / 4];
            if (fail_after_command) {
                throw std::runtime_error("transport failed after device consumed write");
            }
        }
    }
    void complete(yolo26::DisplaySnapshot snapshot) {
        words[0x10 / 4] = snapshot.status;
        words[0x14 / 4] = snapshot.stopped_id;
        words[0x18 / 4] = snapshot.accepted_address;
        words[0x1c / 4] = snapshot.accepted_id;
        words[0x20 / 4] = snapshot.faults;
        ++words[0x28 / 4];
    }
    std::array<std::uint32_t, 20> words{};
    std::vector<std::uint32_t> writes;
    bool fail_after_command = false, tear_snapshot = false;
};

void initialize(yolo26::DisplayController& controller, Registers& registers) {
    controller.initialize();
    require(controller.pending() && !controller.ready() && !controller.poll(), "initial command not in flight");
    registers.complete({0xc1, 0, 0, 0, 0});
    const auto reply = controller.poll();
    require(reply && reply->command == yolo26::DisplayCommand::Initialize && controller.ready() &&
            !controller.must_retain_resources() && registers.words[0x34 / 4] == 0,
            "initial locked/drained snapshot or pattern disable failed");
}

} // namespace

int main() {
    using namespace yolo26;
    Registers registers;
    DisplayController controller(registers);
    initialize(controller, registers);
    rejects([&] { controller.initialize(); });
    DisplayBuffers buffers({{{0x30100000, Compositor720p::kBytes},
                             {0x30500000, Compositor720p::kBytes},
                             {0x30900000, Compositor720p::kBytes}}});
    const auto upload = *buffers.begin_upload({26, 1});
    buffers.finish_upload(upload);
    const auto frame = *buffers.begin_submit();
    require(controller.submit(frame), "START not queued");
    require(registers.words[0x04 / 4] == frame.address && registers.words[0x08 / 4] == frame.hardware_id &&
            registers.words[0x0c / 4] == 1 && !controller.request_snapshot(), "command parameters or exclusivity");
    // START 应答返回执行前的 idle 状态；控制器不把它误判为启动失败或新帧已采用。
    registers.complete({0xc1, 0, 0, 0, 0});
    auto reply = controller.poll();
    require(reply && reply->command == DisplayCommand::Start && reply->accepted() && buffers.occupied() == 1,
            "queue response confused with frame adoption");
    registers.words[0x2c / 4] = 2;
    const auto write_count = registers.writes.size();
    require(!controller.request_snapshot() && registers.writes.size() == write_count,
            "mailbox ready latency caused an early write");
    registers.words[0x2c / 4] = 0;
    require(controller.request_snapshot(), "SNAPSHOT not queued");
    registers.complete({0x142, 0, frame.address, frame.hardware_id, 0});
    reply = controller.poll();
    require(reply && reply->command == DisplayCommand::Snapshot && buffers.observe(reply->snapshot),
            "tagged frame not adopted");
    const auto tag = buffers.begin_stop();
    require(controller.request_stop(tag), "STOP not queued");
    registers.complete({0x142, 0, frame.address, frame.hardware_id, 0});
    reply = controller.poll();
    require(reply && reply->command == DisplayCommand::Stop && buffers.must_retain_resources(),
            "STOP queue response freed a scanning buffer");
    controller.request_snapshot();
    registers.complete({0xc5, tag, frame.address, frame.hardware_id, 8});
    reply = controller.poll();
    require(!buffers.observe_stop(reply->snapshot), "busy reader released after STOP");
    controller.request_snapshot();
    registers.complete({0xc1, tag, frame.address, frame.hardware_id, 8});
    reply = controller.poll();
    require(buffers.observe_stop(reply->snapshot) && !buffers.must_retain_resources(),
            "drained tagged STOP could not close after recorded underflow");

    Registers wrapped;
    wrapped.words[0x24 / 4] = wrapped.words[0x28 / 4] = std::numeric_limits<std::uint32_t>::max();
    DisplayController wrapping(wrapped);
    initialize(wrapping, wrapped);
    require(wrapped.words[0x28 / 4] == 0, "legitimate CSR generation wrap rejected");

    Registers wrong_mode;
    wrong_mode.words[0x44 / 4] = 148500000;
    DisplayController mismatch(wrong_mode);
    rejects([&] { mismatch.initialize(); });
    require(wrong_mode.writes.empty(), "1080p candidate received a DMA command");

    for (unsigned failure = 0; failure < 4; ++failure) {
        Registers broken;
        DisplayController owned(broken);
        initialize(owned, broken);
        if (failure == 0) {
            broken.fail_after_command = true;
            rejects([&] { owned.submit(frame); });
        } else {
            owned.request_snapshot();
            broken.complete({0x142, 0, frame.address, frame.hardware_id, 0});
            if (failure == 1) {
                ++broken.words[0x28 / 4];
            } else if (failure == 2) {
                broken.tear_snapshot = true;
            } else {
                broken.words[0x2c / 4] = 4;
            }
            rejects([&] { owned.poll(); });
        }
        const auto writes_before = broken.writes.size();
        rejects([&] { owned.request_stop(99); });
        require(owned.must_retain_resources() && broken.writes.size() == writes_before,
                "unknown MMIO retried or released its owner");
    }
    std::cout << "PASS display control: pre-command response, adoption, drain, counters and uncertain MMIO\n";
}
