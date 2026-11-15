#pragma once
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace yolo26 {

// 有界、单待处理项交接。消费者取走后持有独立共享引用，生产者不能替换该对象。
// close 丢弃尚未消费项并唤醒等待线程；关闭后不再接受新帧。
template <typename T>
class LatestSlot {
public:
    using Value = std::shared_ptr<const T>;

    bool publish(Value value) {
        if (!value) {
            throw std::invalid_argument("cannot publish a null completed item");
        }
        Value displaced;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return false;
            }
            if (pending_) {
                ++dropped_;
            }
            displaced = std::move(pending_);
            pending_ = std::move(value);
        }
        // 旧对象析构可能归还另一个池的锁，故放到本队列锁之外。
        changed_.notify_one();
        return true;
    }

    Value try_take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::exchange(pending_, {});
    }

    Value wait_take() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return closed_ || pending_; });
        return std::exchange(pending_, {});
    }

    void close() {
        Value discarded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            discarded = std::move(pending_);
        }
        changed_.notify_all();
    }

    std::size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    Value pending_;
    bool closed_ = false;
    std::size_t dropped_ = 0;
};

} // namespace yolo26
