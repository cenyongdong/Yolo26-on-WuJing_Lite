#include "yolo26/npu_trace.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class Action> void rejects(Action action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "unsafe action was accepted");
}

// 邮箱应答故意延迟；STOP 的首次响应也早于最后一箱提交。
class Registers final : public yolo26::RegisterAccess {
public:
    std::array<std::uint32_t, 16> csr{};
    std::vector<std::array<std::uint32_t, 16>> records;
    std::uint32_t status = 0;
    unsigned commands = 0, polls = 0, pending_cycles = 0;
    bool stall = false, foreign_serial = false, late_commit = false;
    Registers() {
        csr[0] = 0x4e505431;
        csr[1] = 0x00010000;
        csr[0x28 / 4] = 512;
        for (unsigned bin = 0; bin < 2; ++bin) {
            std::array<std::uint32_t, 16> words{};
            words[0] = bin;
            words[1] = 10;
            words[2] = 0xbad10001U + bin * 10; // 合法时间戳与硬件错误码相同。
            words[4] = words[6] = words[8] = 1;
            words[10] = 64;
            records.push_back(words);
        }
    }
    std::uint32_t read32(std::uint32_t offset) override {
        ++polls;
        if (offset == 0x18 && pending_cycles && !stall && --pending_cycles == 0) {
            csr[0x18 / 4] = 0;
            csr[0x24 / 4] = csr[0x20 / 4];
        }
        return csr.at(offset / 4);
    }
    void write32(std::uint32_t offset, std::uint32_t value) override {
        require(csr[0x18 / 4] == 0, "write attempted while mailbox pending");
        if (offset != 0x10) {
            csr.at(offset / 4) = value;
            return;
        }
        ++commands;
        csr[0x20 / 4] += foreign_serial ? 2 : 1;
        csr[0x18 / 4] = 1;
        pending_cycles = 3;
        if (value == 1) {
            status = 1;
        } else if (value == 2) {
            late_commit = true;
        } else if (value == 0 && late_commit) {
            status = static_cast<std::uint32_t>((records.size() << 16) | 2);
            late_commit = false;
        }
        csr[0x1c / 4] = value == 3 ? records.at(csr[0x14 / 4] / 16).at(csr[0x14 / 4] % 16) : status;
    }
};

} // namespace

int main() {
    try {
        Registers registers;
        registers.csr[0x20 / 4] = registers.csr[0x24 / 4] = std::numeric_limits<std::uint32_t>::max();
        yolo26::NpuTraceController trace(registers);
        trace.initialize();
        require(registers.csr[0x20 / 4] == 0, "serial did not wrap as uint32");
        const auto commands = registers.commands;
        rejects([&] { trace.start(0, 512); });
        rejects([&] { trace.start(1000001, 512); });
        rejects([&] { trace.start(10, 513); });
        require(registers.commands == commands, "invalid configuration reached hardware");
        trace.start(10, 512);
        rejects([&] { trace.read_frozen(); });
        rejects([&] { trace.start(10, 512); });
        trace.stop();
        const auto capture = trace.read_frozen();
        require(capture.records == registers.records && capture.status == (2U << 16 | 2U),
                "complete frozen records changed");
        trace.start(10, 512);
        trace.stop();
        require(trace.read_frozen().records == registers.records, "next capture could not be read");

        Registers wrong;
        wrong.csr[0] = 0;
        yolo26::NpuTraceController bad_identity(wrong);
        rejects([&] { bad_identity.initialize(); });
        require(wrong.commands == 0, "wrong bitstream received a command");

        Registers active;
        active.status = 1;
        yolo26::NpuTraceController other_owner(active);
        rejects([&] { other_owner.initialize(); });
        require(active.commands == 1, "active capture was stopped or restarted");

        Registers stopped_clock;
        yolo26::NpuTraceController timeout(stopped_clock);
        timeout.initialize();
        stopped_clock.stall = true;
        rejects([&] { timeout.start(10, 512); });
        require(timeout.failed() && stopped_clock.commands == 2 && stopped_clock.csr[0x18 / 4] == 1,
                "timeout cancelled or retried the in-flight command");
        const auto stopped_polls = stopped_clock.polls;
        rejects([&] { timeout.stop(); });
        rejects([&] { timeout.start(10, 512); });
        require(stopped_clock.polls == stopped_polls, "quarantined controller still accessed hardware");

        Registers stolen;
        yolo26::NpuTraceController serial(stolen);
        serial.initialize();
        stolen.foreign_serial = true;
        rejects([&] { serial.start(10, 512); });
        require(serial.failed(), "foreign serial did not quarantine controller");

        Registers corrupt;
        yolo26::NpuTraceController invalid_data(corrupt);
        invalid_data.initialize();
        invalid_data.start(10, 512);
        invalid_data.stop();
        corrupt.records[1][2] += 1;
        rejects([&] { invalid_data.read_frozen(); });
        require(invalid_data.failed(), "record gap was accepted");
        Registers smaller;
        smaller.csr[0x28 / 4] = 256;
        yolo26::NpuTraceController compact(smaller);
        compact.initialize();
        require(compact.capacity() == 256, "compact capacity was ignored");
        rejects([&] { compact.start(80000, 257); });
        compact.start(80000, 256);
        compact.stop();
        require(compact.read_frozen().records == smaller.records, "compact capture changed records");
        std::cout << "PASS: trace protocol, full records, wrap, bounded timeout and failure isolation\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
