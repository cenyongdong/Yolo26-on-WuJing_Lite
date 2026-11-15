#include "yolo26/display_buffers.hpp"
#include "yolo26/compose.hpp"
#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class Action>
void rejects(Action action) {
    bool failed = false;
    try {
        action();
    } catch (const std::logic_error&) {
        failed = true;
    }
    require(failed, "invalid display ownership accepted");
}

std::array<yolo26::DmaRange, 3> memory() {
    return {{{0x30100000, yolo26::Compositor720p::kBytes},
             {0x30500000, yolo26::Compositor720p::kBytes},
             {0x30900000, yolo26::Compositor720p::kBytes}}};
}

yolo26::DisplayUpload upload(yolo26::DisplayBuffers& buffers, std::uint64_t sequence) {
    const auto lease = buffers.begin_upload({17, sequence});
    require(lease.has_value(), "free display slot missing");
    require(buffers.address(*lease) == memory()[lease->slot].address, "upload allocation mismatch");
    buffers.finish_upload(*lease);
    return *lease;
}

yolo26::DisplaySnapshot adopted(const yolo26::DisplaySubmission& frame) {
    return {0x142, 0, frame.address, frame.hardware_id, 0};
}

} // namespace

int main() {
    using namespace yolo26;
    DisplayBuffers buffers(memory());
    require(!buffers.must_retain_resources() && buffers.occupied() == 0, "new ledger owns hardware");
    const auto first = upload(buffers, 1);
    upload(buffers, 2);
    require(buffers.replaced_ready() == 1 && buffers.occupied() == 1, "latest READY not replaced");
    rejects([&] { buffers.address(first); });
    const auto second = *buffers.begin_submit();
    require(second.frame.sequence == 2 && second.start && buffers.must_retain_resources(), "START ownership");
    upload(buffers, 3);
    upload(buffers, 4);
    require(buffers.replaced_ready() == 2 && !buffers.begin_submit(), "hardware-pending frame overwritten");
    require(!buffers.observe({0x52, 0, 0, 0, 0}), "command queue acknowledgement counted as adoption");
    const auto accepted = buffers.observe(adopted(second));
    require(accepted && accepted->sequence == 2 && !buffers.observe(adopted(second)), "adoption repeated");
    const auto fourth = *buffers.begin_submit();
    require(!fourth.start && fourth.frame.sequence == 4, "FLIP target mismatch");
    upload(buffers, 5);
    require(buffers.occupied() == 3 && !buffers.begin_upload({17, 6}), "full triple buffer reused live memory");
    require(!buffers.observe(adopted(second)) && buffers.occupied() == 3, "old scanout freed before adoption");
    buffers.observe(adopted(fourth));
    require(buffers.occupied() == 2, "old scanout not returned after matching adoption");
    upload(buffers, 6);
    require(buffers.replaced_ready() == 3 && buffers.occupied() == 2, "latest frame retained wrong slot");
    rejects([&] { buffers.begin_upload({18, 7}); });
    rejects([&] { buffers.begin_upload({17, 6}); });
    const auto stop = buffers.begin_stop();
    rejects([&] { buffers.begin_stop(); });
    rejects([&] { buffers.begin_submit(); });
    require(!buffers.observe_stop({0xc1, stop - 1, fourth.address, fourth.hardware_id, 0}), "stale STOP accepted");
    require(!buffers.observe_stop({0xc5, stop, fourth.address, fourth.hardware_id, 0}), "busy DMA released");
    require(buffers.observe_stop({0xc1, stop, fourth.address, fourth.hardware_id, 0}) && buffers.stopped() &&
            buffers.occupied() == 0 && !buffers.must_retain_resources(), "tagged drained STOP failed");

    DisplayBuffers interrupted(memory());
    auto unfinished = *interrupted.begin_upload({17, 1});
    require(!interrupted.begin_upload({17, 2}), "multiple upload writers accepted");
    const auto interrupted_stop = interrupted.begin_stop();
    require(!interrupted.observe_stop({0xc1, interrupted_stop, 0, 0, 0}), "unfinished transfer released");
    interrupted.abandon_upload(unfinished);
    require(interrupted.observe_stop({0xc1, interrupted_stop, 0, 0, 0}), "completed cancellation did not release");
    rejects([&] { interrupted.finish_upload(unfinished); });

    // HDMI 欠载后，可信带标签 STOP 仍可证明扫描已停止；不得清错掩盖原故障。
    DisplayBuffers faulty(memory());
    upload(faulty, 1);
    const auto pending = *faulty.begin_submit();
    bool detected = false;
    try {
        faulty.observe({0xc1, 0, 0, 0, 8});
    } catch (const std::runtime_error&) {
        detected = true;
    }
    require(detected && faulty.must_retain_resources(), "HDMI fault not isolated");
    const auto fault_stop = faulty.begin_stop();
    require(faulty.observe_stop({0xc1, fault_stop, 0, 0, 8}), "known HDMI drain could not close");

    // SDK 上传结果未知时，HDMI STOP 只能停止扫描，不能证明 SDK 的 DMA 也停止。
    DisplayBuffers unknown(memory());
    const auto uncertain_upload = *unknown.begin_upload({17, 1});
    unknown.quarantine();
    rejects([&] { unknown.abandon_upload(uncertain_upload); });
    const auto unknown_stop = unknown.begin_stop();
    require(!unknown.observe_stop({0xc1, unknown_stop, 0, 0, 0}) && unknown.must_retain_resources(),
            "unknown SDK transfer freed after unrelated display STOP");

    DisplayBuffers wrong_address(memory());
    upload(wrong_address, 1);
    const auto wanted = *wrong_address.begin_submit();
    auto wrong = adopted(wanted);
    wrong.accepted_address += 64;
    detected = false;
    try {
        wrong_address.observe(wrong);
    } catch (const std::runtime_error&) {
        detected = true;
    }
    require(detected && wrong_address.must_retain_resources(), "matching ID concealed wrong address");
    auto overlapping = memory();
    overlapping[1] = overlapping[0];
    rejects([&] { DisplayBuffers invalid(overlapping); });
    (void)pending;
    std::cout << "PASS display buffers: latest complete frame, three leases, tagged adoption, drain and unknown transfer\n";
}
