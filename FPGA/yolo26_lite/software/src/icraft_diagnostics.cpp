#include "icraft_diagnostics.hpp"
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

namespace yolo26::detail {
namespace {

using Json = nlohmann::json;
using namespace icraft::xrt;

std::map<int, zg330::ZG330Backend> npu_backends(const Session& session) {
    std::map<int, zg330::ZG330Backend> result;
    for (const auto& binding : session.backendBindings()) {
        if (binding.second.is<zg330::ZG330Backend>()) {
            const auto backend = binding.second.cast<zg330::ZG330Backend>();
            result.emplace(backend->zg330backend_id, backend);
        }
    }
    if (result.empty()) {
        throw std::runtime_error("OCM diagnostics require an applied ZG330 backend");
    }
    return result;
}

Json value_description(const zg330::ValueInfo& info) {
    if (!info.defined() || !info->value.defined()) {
        throw std::runtime_error("undefined public NPU value metadata");
    }
    const auto& value = info->value;
    const auto type = value.tensorType();
    Json aliases = Json::array();
    for (const auto& alias : info->real_to) {
        aliases.push_back(alias->v_id);
    }
    return {{"value_id", value->v_id}, {"real", info->real}, {"is_ocm", info->is_ocm},
            {"is_host", info->is_host}, {"storage_bytes", value.storageBytes()},
            {"allocation_bytes", info->byte_size}, {"logical_address", info->logic_addr},
            {"physical_address", info->phy_addr}, {"segment", static_cast<int>(info->segment)},
            {"fake_from", info->fake_from.defined() ? Json(info->fake_from->v_id) : Json(nullptr)},
            {"real_to", aliases}, {"layout", type->layout.toString()},
            {"shape", std::vector<std::int64_t>(type->shape.begin(), type->shape.end())}};
}

Json describe_values(const std::unordered_map<std::int64_t, zg330::ValueInfo>& values) {
    const std::map<std::int64_t, zg330::ValueInfo> ordered(values.begin(), values.end());
    Json result = Json::array();
    for (const auto& entry : ordered) {
        result.push_back(value_description(entry.second));
    }
    return result;
}

Json timer_storage(const Session& session) {
    Json result = Json::array();
    constexpr std::uint64_t maximum_diagnostic_bytes = 1024 * 1024;
    constexpr char digits[] = "0123456789abcdef";
    for (const auto& item : npu_backends(session)) {
        // 使用 SDK 自己拥有的公开 MemChunk；不把计时地址当作普通指针，不猜测 MMU 映射。
        const auto& storage = item.second->time_info;
        Json record = {{"backend_id", item.first}, {"available", storage.defined()}};
        if (storage.defined()) {
            const auto bytes = storage->byte_size;
            record["allocation_bytes"] = bytes;
            record["captured"] = bytes > 0 && bytes <= maximum_diagnostic_bytes;
            if (bytes > 0 && bytes <= maximum_diagnostic_bytes) {
                std::vector<char> raw(static_cast<std::size_t>(bytes));
                storage.read(raw.data(), 0, bytes);
                std::string hex;
                hex.reserve(raw.size() * 2);
                for (const auto byte : raw) {
                    const auto value = static_cast<unsigned char>(byte);
                    hex.push_back(digits[value >> 4]);
                    hex.push_back(digits[value & 15]);
                }
                record["bytes_hex"] = std::move(hex);
                record["encoding"] = "hex_in_increasing_byte_offset";
            }
        }
        result.push_back(std::move(record));
    }
    return result;
}

} // namespace

NpuCounterSample sample_npu_counters(const icraft::xrt::Device& device, unsigned columns) {
    if (columns < 1 || columns > 4) {
        throw std::invalid_argument("counter sampling requires 1..4 physical columns");
    }
    const auto timestamp = [] {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    const auto zg = device.cast<ZG330Device>();
    NpuCounterSample result;
    result.begin_ns = timestamp();
    result.layer_before = zg.layerCount();
    // 生成指令的物理列为 1..cols。保留明确的列编号，不沿用旧报告的 0..cols-1 数组。
    for (unsigned column = 1; column <= columns; ++column) {
        result.reads[column - 1] = zg.rTileCount(static_cast<int>(column));
        result.writes[column - 1] = zg.wTileCount(static_cast<int>(column));
    }
    result.layer_after = zg.layerCount();
    result.end_ns = timestamp();
    return result;
}

nlohmann::json counter_timeline_json(const std::vector<NpuCounterSample>& samples,
                                     unsigned columns, bool truncated) {
    if (columns < 1 || columns > 4) {
        throw std::invalid_argument("counter serialization requires 1..4 physical columns");
    }
    Json rows = Json::array(), ids = Json::array();
    for (unsigned column = 1; column <= columns; ++column) {
        ids.push_back(column);
    }
    for (const auto& sample : samples) {
        rows.push_back({{"begin_ns", sample.begin_ns}, {"end_ns", sample.end_ns},
                        {"layer_before", sample.layer_before}, {"layer_after", sample.layer_after},
                        {"reads", std::vector<std::uint32_t>(sample.reads.begin(), sample.reads.begin() + columns)},
                        {"writes", std::vector<std::uint32_t>(sample.writes.begin(), sample.writes.begin() + columns)}});
    }
    return {{"schema", 1}, {"clock", "host_steady_clock_ns"}, {"column_ids", ids},
            {"samples", rows}, {"truncated", truncated}, {"sleep_between_samples_ns", 200000},
            {"register_reads_are_atomic", false}, {"ddr_bytes_measured", false},
            {"mpe_stall_cycles_measured", false}, {"observed_calls_include_sampling_overhead", true}};
}

void configure_ocm(const icraft::xrt::Session& session, int strategy) {
    if (strategy < -1 || strategy > 3) {
        throw std::invalid_argument("OCM strategy must be -1, 0, 1, 2 or 3");
    }
    // Session 中多个算子可能绑定同一个后端，每个后端仅配置一次，且必须早于 apply。
    for (auto& item : npu_backends(session)) {
        item.second.ocmOptimize(static_cast<zg330::OcmOpt>(strategy));
    }
}

nlohmann::json describe_npu_deployment(const icraft::xrt::Session& session,
                                      const icraft::xrt::Device& device, int strategy) {
    const auto zg = device.cast<ZG330Device>();
    const auto dimensions = zg.getRowsCols();
    Json backends = Json::array();
    for (const auto& item : npu_backends(session)) {
        const auto& backend = item.second;
        Json segments = Json::array();
        // 地址复用/别名意味着这些 byte_size 不能直接相加称为峰值内存或 DDR 通信量。
        for (const auto& entry : backend->phy_segment_map) {
            segments.push_back({{"segment", static_cast<int>(entry.first)},
                                {"bytes", entry.second->byte_size},
                                {"physical_address", entry.second->phy_addr}});
        }
        backends.push_back({{"backend_id", item.first}, {"selected_ocm_strategy", static_cast<int>(backend->ocmopt)},
                            {"merge_hardops", backend->is_mergeHardop}, {"etm_reuse", backend->is_etmOptimize},
                            {"allocation_mode", static_cast<int>(backend->alloc_mode)},
                            {"values", describe_values(backend->forward_info->value_map)},
                            {"planning_values", describe_values(backend->value_info)}, {"segments", segments}});
    }
    return {{"schema", 1}, {"requested_ocm_strategy", strategy}, {"backends", backends},
            {"rows", dimensions.first}, {"cols", dimensions.second},
            {"timer_address", zg->timer_addr}, {"timer_bytes", zg->timer_bytes},
            {"sdk_icore_hz", zg->icore_clk_freq}, {"sdk_ddr_hz", zg->ddr_clk_freq},
            {"clock_values_are_measured", false}, {"session_time_profile", false}};
}

nlohmann::json read_npu_timers(const icraft::xrt::Session& session, const icraft::xrt::Device& device,
                              std::uint32_t layers) {
    auto zg = device.cast<ZG330Device>();
    if (layers == 0 || layers > 10000 || zg->timer_bytes == 0) {
        throw std::runtime_error("native layer timer storage or layer count is unavailable");
    }
    // 调用者必须先证明本次推理完成。此处不 reset、不拆融合组、不重新启动计算。
    const auto times = zg.getLayerTime(static_cast<int>(layers));
    const auto hardware_ms = zg.calcTime();
    const auto dimensions = zg.getRowsCols();
    if (dimensions.second < 1 || dimensions.second > 4) {
        throw std::runtime_error("native MPE column count is outside the documented range");
    }
    Json reads = Json::array(), writes = Json::array(), samples = Json::array(), bits = Json::array();
    bool finite = std::isfinite(hardware_ms) && hardware_ms >= 0;
    for (const auto time : times) {
        std::uint64_t representation = 0;
        static_assert(sizeof(time) == sizeof(representation), "expected IEEE-sized SDK timer result");
        std::memcpy(&representation, &time, sizeof(time));
        bits.push_back(representation);
        samples.push_back(std::isfinite(time) ? Json(time) : Json(nullptr));
        finite = finite && std::isfinite(time) && time >= 0;
    }
    for (int column = 0; column < dimensions.second; ++column) {
        reads.push_back(zg.rTileCount(column));
        writes.push_back(zg.wTileCount(column));
    }
    // 保存 API 原值及 double 位模式；先核验是逐层耗时还是累计时间，再由离线分析解释。
    return {{"schema", 1}, {"expected_layers", layers}, {"api_layer_ms", samples},
            {"api_layer_ms_bits", bits},
            {"api_calc_time_ms", std::isfinite(hardware_ms) ? Json(hardware_ms) : Json(nullptr)},
            {"all_finite_nonnegative", finite}, {"returned_layer_count", times.size()},
            {"read_tile_counts", reads}, {"write_tile_counts", writes},
            {"timer_storage", timer_storage(session)},
            {"tile_counts_are_bytes", false}, {"collected_after_sfb", true},
            {"session_time_profile", false}};
}

} // namespace yolo26::detail
