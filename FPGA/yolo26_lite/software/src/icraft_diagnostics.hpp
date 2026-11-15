#pragma once
#include <icraft-xir/utils/json.hpp>
#include <icraft-xrt/core/session.h>
#include <array>
#include <cstdint>
#include <vector>

namespace yolo26::detail {

// 每次寄存器读取并非原子快照；保留前后时刻及层计数，不能把变化强行归给单层。
struct NpuCounterSample {
    std::uint64_t begin_ns = 0, end_ns = 0;
    std::uint32_t layer_before = 0, layer_after = 0;
    std::array<std::uint32_t, 4> reads{}, writes{};
};

NpuCounterSample sample_npu_counters(const icraft::xrt::Device& device, unsigned columns);
nlohmann::json counter_timeline_json(const std::vector<NpuCounterSample>& samples,
                                     unsigned columns, bool truncated);

// 仅使用 SDK 公开的后端元数据和设备方法，不改写地址、MMU 或时钟。
void configure_ocm(const icraft::xrt::Session& session, int strategy);
nlohmann::json describe_npu_deployment(const icraft::xrt::Session& session,
                                      const icraft::xrt::Device& device, int strategy);
nlohmann::json read_npu_timers(const icraft::xrt::Session& session, const icraft::xrt::Device& device,
                              std::uint32_t layers);

} // namespace yolo26::detail
