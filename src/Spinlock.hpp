#ifndef SIGMOD23ANN_SPINLOCK_HPP
#define SIGMOD23ANN_SPINLOCK_HPP

#include <atomic>

struct Spinlock {
    std::atomic<bool> latch = false;

    inline void lock() {
        bool expected = false;
        while(!latch.compare_exchange_weak(expected, true)) {
            expected = false;
        }
    }

    inline bool try_lock() {
        bool expected = false;
        return latch.compare_exchange_strong(expected, true);
    }

    inline void unlock() {
        bool expected = true;
        latch.compare_exchange_strong(expected, false);
    }
};

#endif //SIGMOD23ANN_SPINLOCK_HPP
