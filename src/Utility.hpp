#ifndef SIGMOD23ANN_UTILITY_HPP
#define SIGMOD23ANN_UTILITY_HPP




template<class T, class TVec = vector<T>>
struct Task {
    TVec& tasks;
    std::atomic<uint64_t> index = 0;

    explicit Task(TVec& tasks): tasks(tasks) {}

    std::optional<T> getTask() {
        auto curr = index.load();
        while (curr < tasks.size()) {
            if (index.compare_exchange_strong(curr, curr + 1)) {
                return { tasks[curr] };
            }
        }
        return {};
    }
};



// [first, second)
vector<Range> splitRange(Range range, uint32_t numRanges) {
    uint32_t size = (range.second - range.first) / numRanges;

    vector<Range> ranges;
    auto start = range.first;
    for (uint32_t i = 0; i < numRanges; i++) {
        uint32_t end = i == numRanges - 1 ? range.second : start + size;
        ranges.emplace_back(start, end);
        start = end;
    }
    return ranges;
}


#endif //SIGMOD23ANN_UTILITY_HPP
