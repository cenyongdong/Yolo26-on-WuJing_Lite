#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolo26 {

struct HardOpBinding {
    std::int64_t operation;
    std::int64_t first_layer;
    std::int64_t layer_count;
    std::vector<std::int64_t> originals;
};

struct CompletionObservation {
    std::vector<bool> output_ready;
    std::uint32_t completed_layers;
    std::uint64_t npu_callbacks;
    std::uint64_t host_compute_callbacks;
};

// 由实际 Session.apply() 后的公开 ZG 元数据构建。融合组数、原始 HardOp 数和硬件 layerCount
// 是三个不同的量；不能将编译图的 399 个 HardOp 直接用作硬件完成门槛。
// 本类不访问 SDK/设备，仅核验绑定覆盖及完成区间；最终读数和 ready 必须来自同一推理调用。
class InferenceCompletion {
public:
    InferenceCompletion(std::vector<std::int64_t> original_hardops,
                        std::vector<std::int64_t> expected_host_ports,
                        std::vector<HardOpBinding> groups,
                        std::vector<std::int64_t> actual_host_ports,
                        std::size_t outputs);
    void validate(const CompletionObservation& observation) const;
    std::uint32_t final_count() const noexcept { return final_count_; }
    std::size_t original_count() const noexcept { return original_count_; }
    const std::vector<HardOpBinding>& groups() const noexcept { return groups_; }

private:
    std::vector<HardOpBinding> groups_;
    std::uint32_t final_count_ = 0;
    std::size_t original_count_ = 0, outputs_ = 0;
};

} // namespace yolo26
