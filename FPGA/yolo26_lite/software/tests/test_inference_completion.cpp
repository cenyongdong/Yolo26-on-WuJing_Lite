#include "yolo26/inference_completion.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

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
    require(rejected, "invalid inference completion was accepted");
}

} // namespace

int main() {
    using namespace yolo26;
    // 三个原始 HardOp 融合为两组，却执行五个硬件层；顺序来自区间而非绑定 map 的键顺序。
    const std::vector<std::int64_t> ops{10, 20, 30}, ports{0, 40};
    const std::vector<HardOpBinding> groups{{30, 3, 2, {}}, {100, 0, 3, {10, 20}}};
    const InferenceCompletion contract(ops, ports, groups, {40, 0}, 1);
    require(contract.original_count() == 3 && contract.groups().size() == 2 && contract.final_count() == 5,
            "node counts substituted for native hardware completion");
    require(contract.groups()[0].operation == 100 && contract.groups()[1].originals[0] == 30,
            "fusion ordering or single-node fallback differs");
    contract.validate({{true}, 5, 2, 0});
    rejects([&] { contract.validate({{true}, 3, 2, 0}); });
    rejects([&] { contract.validate({{true}, 2, 2, 0}); });
    rejects([&] { contract.validate({{true}, 6, 2, 0}); });
    rejects([&] { contract.validate({{false}, 5, 2, 0}); });
    rejects([&] { contract.validate({{true, true}, 5, 2, 0}); });
    rejects([&] { contract.validate({{true}, 5, 0, 0}); });
    rejects([&] { contract.validate({{true}, 5, 2, 1}); });
    rejects([&] { InferenceCompletion missing(ops, ports, {groups[0]}, ports, 1); });
    rejects([&] { InferenceCompletion host(ops, ports, groups, {0, 20}, 1); });
    rejects([&] { InferenceCompletion duplicate_ports(ops, ports, groups, {0, 40, 40}, 1); });
    rejects([&] { InferenceCompletion duplicate_graph({10, 20, 30, 30}, ports, groups, ports, 1); });
    rejects([&] { InferenceCompletion negative_graph({-10, 20, 30}, ports, groups, ports, 1); });
    rejects([&] { InferenceCompletion no_output(ops, ports, groups, ports, 0); });
    for (unsigned failure = 0; failure < 8; ++failure) {
        auto malformed = groups;
        switch (failure) {
        case 0:
            malformed[0].first_layer = 2;
            break;
        case 1:
            malformed[0].first_layer = 4;
            break;
        case 2:
            malformed[0].layer_count = 0;
            break;
        case 3:
            malformed[1].originals.push_back(30);
            break;
        case 4:
            malformed[1].originals.push_back(99);
            break;
        case 5:
            malformed[0].operation = 100;
            break;
        case 6:
            malformed[0].layer_count = std::numeric_limits<std::int64_t>::max();
            break;
        case 7:
            malformed[0].first_layer = -1;
            break;
        }
        rejects([&] { InferenceCompletion invalid(ops, ports, malformed, ports, 1); });
    }
    std::cout << "PASS inference completion: fusion coverage, intervals, overflow and native execution proof\n";
}
