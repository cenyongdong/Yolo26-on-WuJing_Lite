#include "yolo26/control.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
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
    require(rejected, "unsafe control operation accepted");
}

// 模拟同步 RegRegion 的副作用与可注入的传输异常，不连接开发板。
class FakeRegisters final : public yolo26::RegisterAccess {
public:
    std::array<std::uint32_t, 26> words{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> writes;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> maps;
    std::uint32_t pending_opcode = 0, pending_tag = 0;
    int fail_read = -1, fail_write_after = -1;
    unsigned acknowledgements = 0;

    FakeRegisters() { at(0) = 0x59323601; }
    std::uint32_t& at(std::uint32_t offset) { return words.at(offset / 4); }
    std::uint32_t read32(std::uint32_t offset) override {
        if (static_cast<int>(offset) == fail_read) {
            throw std::runtime_error("injected register read failure");
        }
        return at(offset);
    }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        writes.emplace_back(offset, value);
        if (offset == 0x04) {
            require(!(at(0x08) & 15), "mock command overwrote pending transaction");
            pending_opcode = value;
            pending_tag = at(0x10);
            if (value == 2) {
                maps.clear();
            }
            if (value == 3) {
                maps.emplace_back(at(0x34), at(0x38));
            }
            at(0x08) = (at(0x08) & 16) | 5;
            ++at(0x60);
        } else if (offset == 0x5c) {
            require((at(0x08) & 8) && value == at(0x44), "mock ACK before matching response");
            at(0x08) &= 16;
            ++acknowledgements;
        } else if (offset == 0x3c) {
            require(value || !(at(0x08) & 15), "mock STOP released while active");
            at(0x3c) = value;
            at(0x08) = (at(0x08) & ~16U) | (value ? 16U : 0U);
        } else {
            require(!(at(0x08) & 15), "mock parameter overwritten while active");
            at(offset) = value;
        }
        if (static_cast<int>(offset) == fail_write_after) {
            throw std::runtime_error("injected failure after register write took effect");
        }
    }
    void complete(std::uint32_t status = 0, std::uint32_t source_bytes = 0,
                  std::uint32_t destination_bytes = 0) {
        require((at(0x08) & 5) == 5, "mock has no executable command");
        at(0x40) = status;
        at(0x44) = pending_tag;
        at(0x48) = source_bytes;
        at(0x4c) = destination_bytes;
        at(0x50) = 0;
        at(0x54) = 1 | (at(0x3c) ? 4U : 0U);
        at(0x58) = pending_opcode;
        at(0x08) = (at(0x08) & 16) | 9;
        ++at(0x64);
    }
};

yolo26::PreprocessPlan make_plan() {
    return yolo26::make_preprocess_plan({0x31000000, 614400}, 1280, 640, 480,
                                       yolo26::PixelFormat::Yuyv, yolo26::ColorMatrix::Bt601Limited,
                                       {0x32000000, 1228800});
}

void configure(yolo26::PreprocessController& controller, FakeRegisters& registers,
               const yolo26::PreprocessPlan& plan) {
    controller.configure(plan);
    require(registers.pending_opcode == 2, "configuration did not begin with map clear");
    const auto count = plan.horizontal_map.size() + plan.vertical_map.size();
    for (std::size_t step = 0; step <= count; ++step) {
        require(controller.must_retain_resources(), "configuration lost ownership");
        registers.complete();
        const auto result = controller.poll();
        if (step < count) {
            require(!result && registers.pending_opcode == 3, "configuration completed too early");
        } else {
            require(result && result->kind == yolo26::CompletionKind::Configuration && result->succeeded(),
                    "configuration completion missing");
        }
    }
    require(registers.maps.size() == count && !controller.must_retain_resources(), "map upload count");
    for (std::size_t index = 0; index < count; ++index) {
        const auto horizontal = index < plan.horizontal_map.size();
        const auto local = horizontal ? index : index - plan.horizontal_map.size();
        require(registers.maps[index].first == (horizontal ? 0U : 1024U) + local, "map upload order");
        require(registers.maps[index].second == (horizontal ? plan.horizontal_map[local] : plan.vertical_map[local]),
                "map upload value");
    }
}

} // namespace

int main() {
    using namespace yolo26;
    const auto plan = make_plan();
    FakeRegisters registers;
    registers.at(0x60) = registers.at(0x64) = 0xfffffffe;
    PreprocessController controller(registers);
    controller.initialize();
    rejects([&] { controller.start(plan); });
    require(registers.writes.empty(), "unconfigured frame wrote hardware");
    configure(controller, registers, plan);
    require(registers.at(0x60) == 1119 && registers.at(0x64) == 1119, "counter wrap was not preserved");
    auto damaged = plan;
    damaged.vertical_map[0] ^= 1;
    const auto before = registers.writes.size();
    rejects([&] { controller.start(damaged); });
    require(registers.writes.size() == before && controller.phase() == ControlPhase::Idle,
            "invalid plan changed hardware or ownership");
    const auto tag = controller.start(plan);
    require(tag == registers.pending_tag && registers.at(0x14) == plan.source.address &&
            registers.at(0x1c) == (480U << 16 | 640U) && registers.at(0x30) == 80U << 16,
            "frame descriptor register packing");
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        require(!controller.poll() && controller.must_retain_resources(), "pending frame incorrectly released");
    }
    rejects([&] { controller.start(plan); });
    rejects([&] { controller.clear_stop(); });
    registers.complete(0, 614400, 1228800);
    const auto result = controller.poll();
    require(result && result->kind == CompletionKind::Frame && result->tag == tag && result->succeeded() &&
            !controller.must_retain_resources(), "frame completion was not safely released");

    controller.start(plan);
    controller.request_stop();
    require(!controller.poll() && controller.must_retain_resources(), "STOP was treated as immediate cancel");
    registers.complete(5, 128, 64);
    require(controller.poll()->status == 5 && controller.phase() == ControlPhase::Idle, "STOP drain result");
    rejects([&] { controller.start(plan); });
    controller.clear_stop();
    controller.start(plan);
    registers.complete(2, 128, 64);
    registers.at(0x50) = 0x10002;
    require(controller.poll()->status == 2 && !controller.must_retain_resources(), "drained read error ownership");

    // 各类不可信完成均不得 ACK；原始寄存器保持用于调查，内存租约与设备独占必须保留。
    for (unsigned fault = 0; fault < 11; ++fault) {
        FakeRegisters bad;
        PreprocessController guarded(bad);
        guarded.initialize();
        guarded.configure(plan);
        bad.complete();
        if (fault == 0) {
            ++bad.at(0x44);
        }
        if (fault == 1) {
            bad.at(0x54) = 0;
        }
        if (fault == 2) {
            bad.at(0x58) = 3;
        }
        if (fault == 3) {
            bad.at(0x40) = 6;
        }
        if (fault == 4) {
            bad.at(0x50) = 7;
        }
        if (fault == 5) {
            bad.at(0x48) = 8;
        }
        if (fault == 6) {
            ++bad.at(0x64);
        }
        if (fault == 7) {
            bad.at(0x0c) = 1;
        }
        if (fault == 8) {
            bad.at(0x54) |= 8;
        }
        if (fault == 9) {
            bad.fail_read = 0x48;
        }
        if (fault == 10) {
            bad.at(0x08) = 0;
        }
        rejects([&] { guarded.poll(); });
        require(guarded.phase() == ControlPhase::Quarantined && guarded.must_retain_resources() &&
                bad.acknowledgements == 0, "invalid response released or acknowledged");
        rejects([&] { guarded.initialize(); });
        rejects([&] { guarded.poll(); });
    }

    FakeRegisters uncertain;
    PreprocessController guarded(uncertain);
    guarded.initialize();
    uncertain.fail_write_after = 0x04;
    rejects([&] { guarded.configure(plan); });
    require(uncertain.at(0x60) == 1 && guarded.must_retain_resources(), "ambiguous write was retried or freed");
    uncertain.fail_write_after = -1;
    guarded.request_stop();
    require(uncertain.at(0x3c) == 1 && guarded.phase() == ControlPhase::Quarantined,
            "STOP incorrectly recovered an ambiguous transaction");

    FakeRegisters stopped;
    PreprocessController configuring(stopped);
    configuring.initialize();
    configuring.configure(plan);
    configuring.request_stop();
    stopped.complete();
    const auto stopped_result = configuring.poll();
    require(stopped_result && stopped_result->status == 5 && stopped.maps.empty(), "STOP continued map upload");
    configuring.clear_stop();
    rejects([&] { configuring.start(plan); });

    FakeRegisters short_frame;
    PreprocessController frame_check(short_frame);
    frame_check.initialize();
    configure(frame_check, short_frame, plan);
    frame_check.start(plan);
    short_frame.complete(0, 614400, 1228799);
    const auto previous_acks = short_frame.acknowledgements;
    rejects([&] { frame_check.poll(); });
    require(frame_check.must_retain_resources() && short_frame.acknowledgements == previous_acks,
            "short successful frame was published or acknowledged");
    std::cout << "control ownership, map upload, STOP, counter wrap and fault injection passed\n";
}
