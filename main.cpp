#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cassert>
#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>
#include <cstdint>
#include <chrono>
#include <thread>
#include <optional>
#include <mutex>
#include "io.h"


#include <immintrin.h>


using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using Vec = vector<float>;

#define _INT_MAX 2147483640



/**
 * Docs used
 * https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
 * https://en.wikipedia.org/wiki/Locality-sensitive_hashing
 * http://infolab.stanford.edu/~ullman/mining/2009/similarity3.pdf
 * http://infolab.stanford.edu/~ullman/mining/pdf/cs345-lsh.pdf
 * http://infolab.stanford.edu/~ullman/mining/2008/slides/cs345-lsh.pdf
 */

float distance(const Vec &lhs, const Vec &rhs) {
    float ans = 0.0;
    unsigned lensDim = 100;
    for (unsigned i = 0; i < lensDim; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}


float distance128(const Vec &lhs, const Vec &rhs) {
    __m128 sum  = _mm_set1_ps(0);

    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    for (uint32_t i = 0; i < 100; i+=4) {
        __m128 rs = _mm_load_ps(r);
        __m128 ls = _mm_load_ps(l);
        __m128 diff = _mm_sub_ps(ls, rs);
        __m128 prod = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, prod);
        l += 4;
        r += 4;
    }

    float sums[4] = {};
    _mm_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    return ans;
}

float distance256(const Vec &lhs, const Vec &rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(rhs.data() + i);
        __m256 ls = _mm256_load_ps(lhs.data() + i);
        __m256 diff = _mm256_sub_ps(ls, rs);
        __m256 prod = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, prod);
    }

    float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }

    for (unsigned i = 97; i < 100; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }

    return ans;
}

std::default_random_engine rd(123);
vector<float> makeRandoms(size_t len=100) {
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    Vec randVec;
    randVec.reserve(len);
    while (randVec.size() < len) {
        randVec.emplace_back(dist(gen));
    }
    return randVec;
}

float dot(const Vec& lhs, const Vec& rhs) {
    float ans = 0.0;
    for (unsigned i = 0; i < 100; ++i) {
        ans += (lhs[i] * rhs[i]);
    }
    return ans;
}

uint64_t projectRandoms(const vector<Vec>& randVecs, const Vec& values) {
    uint64_t projection = 0;
    for (auto& rv : randVecs) {
        auto dotProduct = dot(values, rv);
        projection <<= 1;
        projection |= (dotProduct > 0);
    }
    return projection; 
}

vector<uint32_t> CalculateOneKnn(const vector<vector<float>> &data,
                                 const vector<uint32_t> &sample_indexes,
                                 const uint32_t id) {
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;
    float lower_bound = _INT_MAX;
    for (unsigned i = 0; i < sample_indexes.size(); i++) {
        uint32_t sample_id = sample_indexes[i];
        if (id == sample_id) continue;  // skip itself.
        float dist = distance128(data[id], data[sample_id]);

        // only keep the top 100
        if (top_candidates.size() < 100 || dist < lower_bound) {
            top_candidates.push(std::make_pair(dist, sample_id));
            if (top_candidates.size() > 100) {
                top_candidates.pop();
            }

            lower_bound = top_candidates.top().first;
        }
    }

    vector<uint32_t> knn;
    while (!top_candidates.empty()) {
        knn.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(knn.begin(), knn.end());
    return knn;
}
void constructResultActual(const vector<Vec>& data, vector<vector<uint32_t>>& result) {

    result.resize(data.size());
    int count = 0;

    vector<uint32_t> sample_indexes(data.size());
    iota(sample_indexes.begin(), sample_indexes.end(), 0);

    auto numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (int i = 0; i < data.size(); ++i) {
        result[i] = CalculateOneKnn(data, sample_indexes, i);
    }



    vector<uint32_t> sizes(101);
    for (uint32_t i=0; i < data.size(); ++i) {
        sizes[result[i].size()]++;
    }
    for (uint32_t i=0; i < sizes.size(); ++i) {
        std::cout << "size: " << i << ", count: " << sizes[i] << std::endl;
    }
}

template<class T>
struct Task {
    vector<T> tasks;
    std::atomic<uint64_t> index = 0;

    explicit Task(vector<T> tasks): tasks(std::move(tasks)) {}

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

template<class K, class V>
vector<K> getKeys(std::unordered_map<K, V>& map) {
    vector<K> keys;
    for(const auto& kv : map) {
        keys.push_back(kv.first);
    }
    return keys;
}


// 2 groups: 0.9281
// 4653
// 5347


// 4 groups: 1.0125
// 2243 + 2410
// 2770 + 2577

// num groups == 1 << 7 == 128 => expected items per group ~ 7000
uint32_t numGroupBits = 0;
void constructResult(const vector<Vec>& data, vector<vector<uint32_t>>& result) {

    std::cout << "expected group size: "  << data.size() / (1<<numGroupBits) << std::endl;

    vector<Vec> randVecs;
    while (randVecs.size() < numGroupBits) {
        auto randv = makeRandoms();
        randVecs.push_back(randv);
    }
    
    std::unordered_map<uint64_t, vector<uint32_t>> groups;

    // load vectors in map
    for (uint32_t i = 0, len=data.size(); i < len; ++i) {
        if (i % 100'000 == 0) std::cout << "inserted: " << i << std::endl;

        uint64_t signature = projectRandoms(randVecs, data[i]);
        if (groups.find(signature) == groups.end()) {
            groups[signature] = { i };
        } else {
            groups[signature].push_back(i);
        }
    }
    std::cout << groups.size() << std::endl;

    result.resize(data.size());

    std::atomic<uint64_t> count = 0;
    auto startTime = high_resolution_clock::now();
    auto start10k = high_resolution_clock::now();

    Task<uint64_t> tasks(getKeys(groups));

    auto numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            std::optional<uint64_t> sig = tasks.getTask();

            while (sig) {
                const auto& group = groups[*sig];
                std::cout << "group size: : " << group.size() << std::endl;
                for (auto& id : group) {
                    result[id] = CalculateOneKnn(data, group, id);

                    auto localCount = count++;
                    if (localCount % 10'000 == 0) {
                        auto currentTime = high_resolution_clock::now();
                        auto durationGroup = duration_cast<milliseconds>(currentTime - start10k);
                        auto durationTotal = duration_cast<milliseconds>(currentTime - startTime);

                        auto percentDone = static_cast<float>(localCount) / data.size();
                        auto estimatedRequiredSec = (durationTotal.count() / percentDone) / 1000;

                        std::cout << "completed: " << localCount << ", 10k time (ms): " << durationGroup.count()
                            << ", total time: " << durationTotal.count() << std::endl;
                        std::cout << "estimated completion time (s): " << estimatedRequiredSec << std::endl;
                        start10k = currentTime;

                    }
                }

                sig = tasks.getTask();
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }

    auto unusedId = 1;
    vector<uint32_t> sizes(101);
    for (uint32_t i=0; i < data.size(); ++i) {
        sizes[result[i].size()]++;
        while (result[i].size() < 100)  {
            result[i].push_back(unusedId);
        }
    }

    for (uint32_t i=0; i < sizes.size(); ++i) {
        std::cout << "size: " << i << ", count: " << sizes[i] << std::endl;
    }
}

int main(int argc, char **argv) {
  string source_path = "dummy-data.bin";

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  // Read data points
  vector<vector<float>> nodes;
  ReadBin(source_path, nodes);


  // Knng constuction
  vector<vector<uint32_t>> knng;
  constructResult(nodes, knng);

  // Save to ouput.bin
  SaveKNNG(knng);

  return 0;
}

