#include "yolo26/frames.hpp"
#include "yolo26/latest_slot.hpp"
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
yolo26::FrameInfo info(std::uint64_t sequence) {
    return {{26, sequence}, sequence + 1, 4, 2, 12, 24, yolo26::PixelFormat::Rgb888};
}
}

int main() {
    using namespace yolo26;
    FramePool pool(3, 24);
    auto first = pool.try_acquire();
    auto second = pool.try_acquire();
    auto third = pool.try_acquire();
    require(!pool.try_acquire() && pool.available() == 0, "pool must be bounded");
    std::memset(first->data(), 0x26, first->capacity());
    auto held = first->publish(info(0));
    auto duplicate = held;
    LatestSlot<FramePacket> queue;
    queue.publish(second->publish(info(1)));
    queue.publish(third->publish(info(2)));
    require(queue.dropped() == 1 && pool.available() == 1, "only pending frame may be replaced");
    auto consumer = queue.try_take();
    require(consumer->info().key.sequence == 2 && !queue.try_take(), "latest complete frame selected");
    held.reset();
    require(pool.available() == 1 && duplicate->data()[0] == 0x26, "held reader cannot be overwritten");
    duplicate.reset();
    consumer.reset();
    require(pool.available() == 3, "last consumer returns slots");
    {
        auto incomplete = pool.try_acquire();
        auto bad = info(9);
        bad.bytes_used = 3;
        bool rejected = false;
        try { incomplete->publish(bad); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && pool.available() == 2, "failed publish retains writer");
    }
    require(pool.available() == 3, "abandoned fill returned");
    FrameRef survivor;
    {
        FramePool temporary(1, 24);
        survivor = temporary.try_acquire()->publish(info(10));
    }
    require(survivor->info().key.sequence == 10, "frame outlives pool wrapper");

    // 一个生产者和一个慢消费者反复覆盖待处理项；持有帧的字节不能改变。
    LatestSlot<FramePacket> concurrent;
    FramePool threaded_pool(6, 24);
    std::atomic<bool> bad_frame{false};
    std::atomic<unsigned> seen{0};
    std::thread reader([&] {
        std::uint64_t last = 0;
        while (auto frame = concurrent.wait_take()) {
            const auto sequence = frame->info().key.sequence;
            if (sequence <= last) bad_frame = true;
            last = sequence;
            std::this_thread::yield();
            for (std::size_t index = 0; index < frame->info().bytes_used; ++index) {
                if (frame->data()[index] != static_cast<std::uint8_t>(sequence)) bad_frame = true;
            }
            ++seen;
        }
    });
    for (std::uint64_t sequence = 1; sequence <= 10000; ++sequence) {
        auto writable = threaded_pool.try_acquire();
        require(bool(writable), "six slots cover one reader and one pending frame");
        std::memset(writable->data(), static_cast<std::uint8_t>(sequence), writable->capacity());
        concurrent.publish(writable->publish(info(sequence)));
    }
    concurrent.close();
    reader.join();
    require(!bad_frame && seen > 0 && threaded_pool.available() == 6, "concurrent ownership integrity");
    require(!concurrent.wait_take(), "closed waiter wakes without work");
    require(!concurrent.publish(survivor), "closed queue refuses new work");
    std::cout << "PASS frames: fixed pool, immutable references, latest queue, close; observed=" << seen << '\n';
}
