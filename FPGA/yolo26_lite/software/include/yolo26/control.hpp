#pragma once
#include "yolo26/preprocess.hpp"
#include <cstdint>
#include <optional>

namespace yolo26 {

// 仅接收外设内字节偏移；适配层持有基地址、公开 SDK RegRegion 与设备独占。
// 读写必须同步且保序；不得使用私有映射或猜测 ADDR/BOTH 指针。
class RegisterAccess {
public:
    virtual ~RegisterAccess() = default;
    virtual std::uint32_t read32(std::uint32_t offset) = 0;
    virtual void write32(std::uint32_t offset, std::uint32_t value) = 0;
};

enum class ControlPhase { Uninitialized, Idle, Configuring, Running, Quarantined };
enum class CompletionKind { Configuration, Frame };

struct ControlCompletion {
    CompletionKind kind;
    std::uint32_t tag;
    std::uint32_t status;
    std::uint32_t source_bytes;
    std::uint32_t destination_bytes;
    std::uint32_t stage_status;
    std::uint32_t flags;
    bool succeeded() const noexcept { return status == 0; }
};

// 单线程、单所有者。类本身不分配/释放 DMA 内存，析构也不做 MMIO 或假装停止硬件。
// 调用者必须在 configure/start 到 poll 返回有效完成前保留设备锁、源/目标物理租约。
// 时间预算由外层负责；poll 返回 nullopt 只表示仍在途，不能据此回收资源。
class PreprocessController {
public:
    explicit PreprocessController(RegisterAccess& registers) noexcept;
    PreprocessController(const PreprocessController&) = delete;
    PreprocessController& operator=(const PreprocessController&) = delete;

    void initialize();
    void configure(const PreprocessPlan& plan);
    std::uint32_t start(const PreprocessPlan& plan);
    std::optional<ControlCompletion> poll();
    void request_stop();
    void clear_stop();

    ControlPhase phase() const noexcept { return phase_; }
    bool must_retain_resources() const noexcept;
    bool stop_requested() const noexcept { return stop_requested_; }

private:
    std::uint32_t read(std::uint32_t offset);
    void write(std::uint32_t offset, std::uint32_t value);
    [[noreturn]] void quarantine(const char* message);
    void require_idle() const;
    void check_hardware_idle();
    void issue(std::uint32_t opcode);
    void issue_map_entry();
    void validate_response(const ControlCompletion& result, std::uint32_t opcode);

    RegisterAccess& registers_;
    ControlPhase phase_ = ControlPhase::Uninitialized;
    bool stop_requested_ = false;
    std::uint32_t next_tag_ = 1;
    std::uint32_t pending_tag_ = 0;
    std::uint32_t pending_opcode_ = 0;
    std::uint32_t completed_count_ = 0;
    std::uint32_t expected_source_bytes_ = 0;
    std::uint32_t expected_destination_bytes_ = 0;
    std::optional<PreprocessPlan> configuration_;
    std::optional<Letterbox> configured_geometry_;
    std::size_t configuration_step_ = 0;
};

} // namespace yolo26
