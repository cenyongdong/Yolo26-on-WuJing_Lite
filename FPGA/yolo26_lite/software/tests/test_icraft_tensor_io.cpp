#include "../src/icraft_tensor_io.hpp"
#include "yolo26/postprocess.hpp"
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void rejects(Function action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "invalid direct transfer was accepted");
}

} // namespace

int main() {
    try {
        namespace xir = icraft::xir;
        namespace xrt = icraft::xrt;
        using yolo26::detail::validate_plain_fp32_output;
        using yolo26::detail::read_plain_fp32_output;
        using yolo26::detail::wrap_host_fp32_input;
        const auto geometry = yolo26::kRectangularModel;
        const auto input_count = geometry.rgb_elements();
        std::vector<float> input_storage(input_count, 0.0F), input_result(input_count);
        const xir::TensorType input_type(xir::FloatType::FP32(), {1, 384, 512, 3}, xir::Layout("NHWC"));
        {
            auto input = wrap_host_fp32_input(input_type, input_storage.data(), input_count, geometry);
            for (std::size_t i = 0; i < input_count; ++i) {
                input_storage[i] = static_cast<float>(i % 256) / 255.F;
            }
            require(input.isReady(), "wrapped Host input did not become ready");
            input.read(reinterpret_cast<char*>(input_result.data()), 0, input_count * sizeof(float));
            require(std::memcmp(input_storage.data(), input_result.data(), input_count * sizeof(float)) == 0,
                    "wrapped Host input did not expose updated bytes");
        }
        require(input_storage == input_result, "Host Tensor tore down the application buffer");
        rejects([&] { wrap_host_fp32_input(input_type, input_storage.data(), input_count - 1, geometry); });
        rejects([&] { wrap_host_fp32_input(input_type, nullptr, input_count, geometry); });
        rejects([&] { wrap_host_fp32_input(input_type, input_storage.data(), input_count, yolo26::kSquareModel); });
        const auto count = yolo26::kRawElements;
        xir::TensorType type(xir::FloatType::FP32(), {1, 84, 8400}, xir::Layout("**C"));
        type.createAddMergedDistr({2});
        validate_plain_fp32_output(type);
        // 大缓冲的非零偏移视图：SDK read 必须遵从 Tensor offset，而不是猜测 chunk 起点。
        xrt::Tensor backing(xir::TensorType(xir::FloatType::FP32(),
                            {static_cast<std::int64_t>(count + 16)}, xir::Layout("C")));
        backing.mallocOn(xrt::HostDevice::MemRegion());
        xrt::Tensor tensor(type, backing.chunk(), 16 * sizeof(float));
        std::vector<float> source(count), actual(count + 2, 123.5F);
        for (std::size_t i = 0; i < count; ++i) {
            source[i] = static_cast<float>(static_cast<int>((i * 37) % 32768) - 16384) / 128.F;
        }
        source[0] = -0.0F;
        source[count - 1] = 1.0F / 8192.F;
        tensor.write(0, reinterpret_cast<char*>(source.data()), count * sizeof(float));
        read_plain_fp32_output(tensor, actual.data() + 1, count);
        require(actual.front() == 123.5F && actual.back() == 123.5F, "destination guard overwritten");
        require(std::memcmp(source.data(), actual.data() + 1, count * sizeof(float)) == 0,
                "direct transfer changed FP32 bytes or offset");
        std::ostringstream serialized(std::ios::out | std::ios::binary);
        tensor.dump(serialized, "SFB");
        const auto bytes = serialized.str();
        require(bytes.size() == count * sizeof(float) &&
                std::memcmp(bytes.data(), actual.data() + 1, bytes.size()) == 0,
                "direct transfer differs from official SFB");
        rejects([&] { read_plain_fp32_output(tensor, actual.data(), count - 1); });
        rejects([&] { read_plain_fp32_output(tensor, nullptr, count); });
        rejects([&] { validate_plain_fp32_output(xir::TensorType(xir::FloatType::FP16(),
                                        {1, 84, 8400}, xir::Layout("**C"))); });
        rejects([&] { validate_plain_fp32_output(xir::TensorType(xir::FloatType::FP32(),
                                        {1, 8400, 84}, xir::Layout("**C"))); });
        rejects([&] { validate_plain_fp32_output(xir::TensorType(xir::FloatType::FP32(),
                                        {1, 84, 8400}, xir::Layout("NCH"))); });
        xir::TensorType padded(xir::FloatType::FP32(), {1, 84, 8400}, xir::Layout("**C"));
        padded.createAddMergedDistr({2}).setUnvalid(8399, 8400);
        rejects([&] { validate_plain_fp32_output(padded); });
        std::cout << "Host Tensor buffer lifetime, offset/read/SFB identity and invalid layouts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
