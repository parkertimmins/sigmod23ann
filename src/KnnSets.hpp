#ifndef SIGMOD23ANN_KNNSETS_HPP
#define SIGMOD23ANN_KNNSETS_HPP

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <algorithm>
#include <queue>
#include <vector>
#include <utility>
#include "Constants.hpp"
#include "LinearAlgebra.hpp"
#include "Spinlock.hpp"

using std::pair;
using std::vector;
using std::queue;


struct KnnSetScannableSimd {
public:
    alignas(sizeof(__m256)) float dists[100] = {};
    alignas(sizeof(__m256)) uint32_t current_ids[100] = {};
    uint32_t size = 0;
    float lower_bound = 0; // 0 -> max val in first 100 -> decreases
    bool contains(uint32_t node) {
        for (uint32_t i = 0; i < size; ++i) {
            if (current_ids[i] == node) { return true; }
        }
        return false;
    }

    bool containsFull(uint32_t node) {
        __m256i pattern = _mm256_set1_epi32(node);
        auto* ids = current_ids;
        for (uint32_t i = 0; i < 96; i+=8) {
            auto block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids));
            auto match = _mm256_movemask_epi8(_mm256_cmpeq_epi32(pattern, block));
            if (match) { return true; }
            ids += 8;
        }
        for (uint32_t i = 96; i < size; ++i) {
            if (current_ids[i] == node) { return true; }
        }
        return false;
    }

    void append(const uint32_t candidate_id, float dist) {
        current_ids[size] = candidate_id;
        dists[size] = dist;
        size++;
    }

    // find index of distance that is known to exist
    uint32_t getIdxOfDist(float matchDist) {
        __m256i pattern = _mm256_castps_si256(_mm256_set1_ps(matchDist));
        auto* distances = dists;
        for (uint32_t i = 0; i < 96; i+=8) {
            __m256i block = _mm256_castps_si256(_mm256_load_ps(distances));
            uint32_t match = _mm256_movemask_epi8(_mm256_cmpeq_epi32(pattern, block));
            if (match) {
                return i + (__builtin_ctz(match) >> 2);
            }
            distances += 8;
        }
        for (uint32_t i = 96; i < size; ++i) {
            if (dists[i] == matchDist) { return i; }
        }

        __builtin_unreachable();
    }

    float getMax() {
        __m256 maxes = _mm256_set1_ps(std::numeric_limits<float>::min());
        auto* distances = dists;
        for (uint32_t i = 0; i < 96; i+=8) {
            __m256 block = _mm256_load_ps(distances);
            maxes = _mm256_max_ps(maxes, block);
            distances += 8;
        }

        alignas(sizeof(__m256)) float maxArr[8] = {};
        _mm256_store_ps(maxArr, maxes);
        float max = std::numeric_limits<float>::min();
        for (float m: maxArr) {
            max = std::max(m, max);
        }
        for (unsigned i = 96; i < dims; ++i) {
            max = std::max(dists[i], max);
        }
        return max;
    }

    // This may misorder nodes of equal sizes
    void addCandidate(const uint32_t candidate_id, float dist) {
        if (size < k) {
            if (!contains(candidate_id)) {
                append(candidate_id, dist);
                lower_bound = std::max(lower_bound, dist);
            }
        } else if (dist < lower_bound) {
            if (!containsFull(candidate_id)) {
                uint32_t maxIdx = getIdxOfDist(lower_bound);
                dists[maxIdx] = dist;
                current_ids[maxIdx] = candidate_id;
                lower_bound = getMax();
            }
        }
    }

    vector<uint32_t> finalize() {
        vector<pair<float, uint32_t>> queue;
        queue.reserve(size);
        for (uint32_t i = 0; i < size; ++i) {
            queue.emplace_back(dists[i], current_ids[i]);
        }

        std::sort(queue.begin(), queue.begin() + size);
        vector<uint32_t> knn;
        for (uint32_t i = 0; i < size; ++i) {
            knn.push_back(queue[i].second);
        }
        return knn;
    }
};

struct KnnSetSorted {
public:
    vector<pair<float, uint32_t>> queue;
    float lower_bound = 0; // 0 -> max val in first 100 -> decreases

    KnnSetSorted() { queue.reserve(k); }

    bool contains(uint32_t node) {
        for (auto& [dist, id] : queue) {
            if (id == node) { return true; }
        }
        return false;
    }

    void append(pair<float, uint32_t> nodePair) {
        queue.push_back(nodePair);
    }

    // This may misorder nodes of equal sizes
    void addCandidate(const uint32_t candidate_id, float dist) {
        if (queue.size() < k) {
            if (!contains(candidate_id)) {
                append({dist, candidate_id});
            }
            if (queue.size() == k) {
                std::sort(queue.begin(), queue.end(), std::ranges::greater{});
                lower_bound = queue[0].first;
            }
        } else if (dist < lower_bound) {
            for (uint32_t i = 0; i < 99; ++i) {
                auto& [otherDist, id] = queue[i];
                if (id == candidate_id) {
                    return;
                } else if (dist <= otherDist) {
                    queue[i] = {dist, candidate_id};
                    return;
                } else {
                    queue[i] = queue[i+1];
                }
            }
            if (queue[99].second != candidate_id) {
                queue[99] = {dist, candidate_id};
            }
            lower_bound = queue[0].first;
        }
    }

    vector<uint32_t> finalize() {
        std::sort(queue.begin(), queue.end());
        vector<uint32_t> knn;
        for (auto& [dist, id] : queue) {
            knn.push_back(id);
        }
        return knn;
    }
};




struct KnnSetScannable {
public:
    vector<pair<float, uint32_t>> queue;
    uint32_t size = 0;
    float lower_bound = 0; // 0 -> max val in first 100 -> decreases

    KnnSetScannable() {
        queue.resize(k);
    }

    bool contains(uint32_t node) {
        for (uint32_t i = 0; i < size; ++i) {
            auto id = queue[i].second;
            if (id == node) { return true; }
        }
        return false;
    }

    void append(pair<float, uint32_t> nodePair) {
        queue[size] = std::move(nodePair);
        size++;
    }

    // This may misorder nodes of equal sizes
    void addCandidate(const uint32_t candidate_id, float dist) {
        if (size < k) {
            if (!contains(candidate_id)) {
                append({dist, candidate_id});
                lower_bound = std::max(lower_bound, dist);
            }
        } else if (dist < lower_bound) {
            float secondMaxVal = std::numeric_limits<float>::min();
            float maxVal = std::numeric_limits<float>::min();
            uint32_t maxIdx = -1;
            for (uint32_t i = 0; i < size; ++i) {
                auto& [otherDist, id] = queue[i];
                if (id == candidate_id) { return; }
                if (otherDist > maxVal) {
                    secondMaxVal = maxVal;
                    maxVal = otherDist;
                    maxIdx = i;
                } else if (otherDist > secondMaxVal) {
                    secondMaxVal = otherDist;
                }
            }

            queue[maxIdx] = {dist, candidate_id};
            lower_bound = std::max(secondMaxVal, dist);
        }
    }

    void merge(vector<pair<float, uint32_t>>& left,
               vector<pair<float, uint32_t>>& right,
               vector<pair<float, uint32_t>>& output) {

        // l and r point to next items to insert
        uint32_t l = 0;
        uint32_t r = 0;

        // out points to next insert point
        uint32_t out = 0;

        auto rightSize = size;
        if (!left.empty() && rightSize > 0) {
            if (left[l] <= right[r]) {
                output[out++] = left[l++];
            } else {
                output[out++] = right[r++];
            }
        } else if (!left.empty()) {
            output[out++] = left[l++];
        } else if (rightSize > 0) {
            output[out++] = right[r++];
        } else {
            size = out;
            return;
        }

        while (out < k && (l < left.size() || r < rightSize)) {
            if (l < left.size() && r < rightSize) {
                if (left[l] == output[out - 1]) {
                    l++;
                } else if (right[r] == output[out - 1]) {
                    r++;
                } else if (left[l] <= right[r]) {
                    output[out++] = left[l++];
                } else {
                    output[out++] = right[r++];
                }
            } else if (l < left.size()) {
                while (out < k && l < left.size())  {
                    if (left[l] == output[out - 1]) {
                        l++;
                    } else {
                        output[out++] = left[l++];
                    }
                }
            } else {
                while (out < k && r < rightSize)  {
                    if (right[r] == output[out - 1]) {
                        r++;
                    } else {
                        output[out++] = right[r++];
                    }
                }
            }
        }
        size = out;
    }

    void mergeCandidates(vector<pair<float, uint32_t>>& distPairCache, vector<pair<float, uint32_t>>& outQueue) {
        merge(distPairCache, queue, outQueue);
        std::swap(queue, outQueue);
    }

    vector<uint32_t> finalize() {
        std::sort(queue.begin(), queue.begin() + size);
        vector<uint32_t> knn;
        for (uint32_t i = 0; i < size; ++i) {
            knn.push_back(queue[i].second);
        }
        return knn;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////
// KnnSet related functions below
///////////////////////////////////////////////////////////////////////////////////////////////
template<class TKnnSet>
void addCandidates(float points[][104],
                   vector<uint32_t>& indices,
                   Range range,
                   vector<TKnnSet>& idToKnn) {
    for (uint32_t i=range.first; i < range.second-1; ++i) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];
        for (uint32_t j=i+1; j < range.second; ++j) {
            auto id2 = indices[j];
            float dist = distance(points[id1], points[id2]);
            knn1.addCandidate(id2, dist);
            idToKnn[id2].addCandidate(id1, dist);
        }
    }
}

vector<uint32_t> padResult(uint32_t numPoints, vector<vector<uint32_t>>& result) {
    auto unusedId = 1;
    vector<uint32_t> sizes(101);
    for (uint32_t i=0; i < numPoints; ++i) {
        sizes[result[i].size()]++;
        while (result[i].size() < dims)  {
            result[i].push_back(unusedId);
        }
    }
    return sizes;
}


void addCandidatesGroup(float points[][104],
                        vector<uint32_t>& group,
                        vector<KnnSetScannable>& idToKnn) {
    uint32_t groupSize = group.size();
    for (uint32_t i = 0; i < groupSize-1; ++i) {
        auto id1 = group[i];
        auto& knn1 = idToKnn[id1];
        for (uint32_t j=i+1; j < groupSize; ++j) {
            auto id2 = group[j];
            float dist = distance(points[id1], points[id2]);
            knn1.addCandidate(id2, dist);
            idToKnn[id2].addCandidate(id1, dist);
        }
    }
}


#endif //SIGMOD23ANN_KNNSETS_HPP
