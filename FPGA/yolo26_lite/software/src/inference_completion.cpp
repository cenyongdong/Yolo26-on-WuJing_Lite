#include "yolo26/inference_completion.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace yolo26 {
namespace {

std::set<std::int64_t> unique_ids(const std::vector<std::int64_t>& values) {
    std::set<std::int64_t> unique;
    for (const auto value : values) {
        if (value < 0 || !unique.insert(value).second) {
            throw std::invalid_argument("negative or repeated graph operation identity");
        }
    }
    return unique;
}

} // namespace

InferenceCompletion::InferenceCompletion(std::vector<std::int64_t> original_hardops,
                                         std::vector<std::int64_t> expected_host_ports,
                                         std::vector<HardOpBinding> groups,
                                         std::vector<std::int64_t> actual_host_ports,
                                         std::size_t outputs)
    : groups_(std::move(groups)), original_count_(original_hardops.size()), outputs_(outputs) {
    if (original_hardops.empty() || expected_host_ports.empty() || groups_.empty() || !outputs) {
        throw std::invalid_argument("empty inference graph completion contract");
    }
    const auto originals = unique_ids(original_hardops);
    const auto expected_ports = unique_ids(expected_host_ports);
    if (expected_ports != unique_ids(actual_host_ports)) {
        throw std::invalid_argument("actual Host input/output bindings differ from the compiled graph");
    }
    for (const auto port : expected_ports) {
        if (originals.count(port)) {
            throw std::invalid_argument("graph port also appears as a HardOp");
        }
    }
    std::set<std::int64_t> group_ids, covered;
    for (auto& group : groups_) {
        if (group.operation < 0 || expected_ports.count(group.operation) ||
            !group_ids.insert(group.operation).second || group.first_layer < 0 || group.layer_count <= 0) {
            throw std::invalid_argument("invalid fused operation identity or layer interval");
        }
        if (group.originals.empty()) {
            group.originals.push_back(group.operation);
        }
        for (const auto original : group.originals) {
            if (!originals.count(original) || !covered.insert(original).second) {
                throw std::invalid_argument("fusion group repeats or invents an original HardOp");
            }
        }
    }
    if (covered != originals) {
        throw std::invalid_argument("fusion bindings do not cover every compiled HardOp");
    }
    std::sort(groups_.begin(), groups_.end(), [](const auto& left, const auto& right) {
        return left.first_layer < right.first_layer;
    });
    std::uint64_t next = 0;
    for (const auto& group : groups_) {
        if (static_cast<std::uint64_t>(group.first_layer) != next ||
            static_cast<std::uint64_t>(group.layer_count) > std::numeric_limits<std::uint32_t>::max() - next) {
            throw std::invalid_argument("completion intervals overlap, skip layers or overflow the device counter");
        }
        next += static_cast<std::uint64_t>(group.layer_count);
    }
    final_count_ = static_cast<std::uint32_t>(next);
}

void InferenceCompletion::validate(const CompletionObservation& observation) const {
    if (observation.output_ready.size() != outputs_ ||
        !std::all_of(observation.output_ready.begin(), observation.output_ready.end(), [](bool value) {
            return value;
        }) || observation.completed_layers != final_count_ || !observation.npu_callbacks ||
        observation.host_compute_callbacks != 0) {
        throw std::runtime_error("native outputs, layer completion or actual backend execution are not proven");
    }
}

} // namespace yolo26
