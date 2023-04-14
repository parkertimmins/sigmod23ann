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
#include <tsl/robin_set.h>
#include "oneapi/tbb.h"
#include <tuple>

using std::pair;
using std::vector;
using std::queue;
using std::tuple;



struct alignas(64) KnnSetScannableSimd {
public:
    alignas(sizeof(__m256)) float dists[104] = { 0 };
    alignas(sizeof(__m256)) uint32_t current_ids[100] = {};
    uint32_t size = 0;
    uint32_t lowerBoundIdx = -1;
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

    uint32_t append(const uint32_t candidate_id, float dist) {
        auto idx = size;
        current_ids[idx] = candidate_id;
        dists[idx] = dist;
        size++;
        return idx;
    }

    uint32_t getMaxIdx() {
        auto* distances = dists;
        __m256 maxes = _mm256_load_ps(distances);
        __m256i maxIndices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i currIndices = _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8);
        __m256i inc = _mm256_set1_epi32(8);
        distances+=8;
        for (uint32_t i = 8; i < 104; i+=8) {
            __m256 block = _mm256_load_ps(distances);
            __m256i gt = _mm256_castps_si256(_mm256_cmp_ps(block, maxes, _CMP_GT_OS));
            maxIndices = _mm256_blendv_epi8(maxIndices, currIndices, gt);
            maxes = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(maxes), _mm256_castps_si256(block), gt));
            currIndices = _mm256_add_epi32(currIndices, inc);
            distances += 8;
        }

        alignas(sizeof(__m256)) float maxArr[8] = {};
        uint32_t maxIdxArr[8] = {};
        _mm256_store_ps(maxArr, maxes);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(maxIdxArr), maxIndices);
        float max = std::numeric_limits<float>::min();
        uint32_t maxIdx = -1;
        for (uint32_t i = 0; i < 8; ++i) {
            if (maxArr[i] > max) {
                max = maxArr[i];
                maxIdx = maxIdxArr[i];
            }
        }
        return maxIdx;
    }

    bool addCandidateLessThan(float& lower_bound, const uint32_t candidate_id, float dist) {
        if (size < k) {
            if (!contains(candidate_id)) {
                append(candidate_id, dist);
                if (size == k) {
                    lowerBoundIdx = getMaxIdx();
                    lower_bound = dists[lowerBoundIdx];
                }
                return true;
            }
        } else if (!containsFull(candidate_id)) {
            dists[lowerBoundIdx] = dist;
            current_ids[lowerBoundIdx] = candidate_id;
            lowerBoundIdx = getMaxIdx();
            lower_bound = dists[lowerBoundIdx];
            return true;
        }
        return false;
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




struct alignas(64) KnnSetInline {
public:
    pair<float, uint32_t> queue[100];
    uint32_t size = 0;
    float lower_bound = 0; // 0 -> max val in first 100 -> decreases

    bool contains(uint32_t node) {
        for (uint32_t i = 0; i < size; ++i) {
            auto id = get<1>(queue[i]);
            if (id == node) { return true; }
        }
        return false;
    }

    void append(float dist, uint32_t id) {
        queue[size] = std::make_pair(dist, id);
        size++;
    }

    // This may misorder nodes of equal sizes
    bool addCandidate(const uint32_t candidate_id, float dist) {
        if (size < k) {
            if (!contains(candidate_id)) {
                append(dist, candidate_id);
                lower_bound = std::max(lower_bound, dist);
                return true;
            }
            return false;
        } else if (dist < lower_bound) {
            float secondMaxVal = std::numeric_limits<float>::min();
            float maxVal = std::numeric_limits<float>::min();
            uint32_t maxIdx = -1;
            for (uint32_t i = 0; i < size; ++i) {
                auto& [otherDist, id] = queue[i];
                if (id == candidate_id) { return false; }
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
            return true;
        }
        return false;
    }

    vector<uint32_t> finalize() {
        std::sort(queue, queue + size);
        vector<uint32_t> knn;
        for (uint32_t i = 0; i < size; ++i) {
            knn.push_back(get<1>(queue[i]));
        }
        return knn;
    }
};

//static_assert(sizeof(KnnSetInline) % 64 == 0);

struct KnnSetScannable {
public:
    vector<tuple<float, uint32_t, bool>> queue;
    uint32_t size = 0;
    float lower_bound = 0; // 0 -> max val in first 100 -> decreases

    KnnSetScannable() {
        queue.resize(k);
    }

    bool contains(uint32_t node) {
        for (uint32_t i = 0; i < size; ++i) {
            auto id = get<1>(queue[i]);
            if (id == node) { return true; }
        }
        return false;
    }

    void append(float dist, uint32_t id, bool isNew) {
        queue[size] = std::make_tuple(dist, id, isNew);
        size++;
    }

    // This may misorder nodes of equal sizes
    bool addCandidate(const uint32_t candidate_id, float dist) {
        if (size < k) {
            if (!contains(candidate_id)) {
                append(dist, candidate_id, true);
                lower_bound = std::max(lower_bound, dist);
                return true;
            }
            return false;
        } else if (dist < lower_bound) {
            float secondMaxVal = std::numeric_limits<float>::min();
            float maxVal = std::numeric_limits<float>::min();
            uint32_t maxIdx = -1;
            for (uint32_t i = 0; i < size; ++i) {
                auto& [otherDist, id, otherIsNew] = queue[i];
                if (id == candidate_id) { return false; }
                if (otherDist > maxVal) {
                    secondMaxVal = maxVal;
                    maxVal = otherDist;
                    maxIdx = i;
                } else if (otherDist > secondMaxVal) {
                    secondMaxVal = otherDist;
                }
            }

            queue[maxIdx] = {dist, candidate_id, true};
            lower_bound = std::max(secondMaxVal, dist);
            return true;
        }
        return false;
    }

    vector<uint32_t> finalize() {
        std::sort(queue.begin(), queue.begin() + size);
        vector<uint32_t> knn;
        for (uint32_t i = 0; i < size; ++i) {
            knn.push_back(get<1>(queue[i]));
        }
        return knn;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////
// KnnSet related functions below
///////////////////////////////////////////////////////////////////////////////////////////////
template<class TKnnSet>
void addCandidates(float points[][112],
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

template<class TKnnSet>
void addCandidatesCopy(
                   float points[][112],
                   float pointsCopy[][112],
                   vector<uint32_t>& indices,
                   Range range,
                   vector<TKnnSet>& idToKnn) {

    for (uint32_t i=range.first; i < range.second; ++i) {
        std::memcpy(pointsCopy[i], points[indices[i]], 100 * sizeof(float));
    }
    for (uint32_t i=range.first; i < range.second-1; ++i) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];
        for (uint32_t j=i+1; j < range.second; ++j) {
            auto id2 = indices[j];
            float dist = distance(pointsCopy[i], pointsCopy[j]);
            knn1.addCandidate(id2, dist);
            idToKnn[id2].addCandidate(id1, dist);
        }
    }
}

void addCandidatesLessThan(
        float points[][112],
        vector<uint32_t>& group,
        vector<float>& bounds,
        vector<KnnSetScannableSimd>& idToKnn) {

    uint32_t groupSize = group.size();
    vector<float[112]> pointsCopy(groupSize);
    for (uint32_t i=0; i < groupSize; ++i) {
        std::memcpy(pointsCopy[i], points[group[i]], 100 * sizeof(float));
    }
    for (uint32_t i = 0; i < groupSize-1; ++i) {
        auto id1 = group[i];
        auto& knn1 = idToKnn[id1];
        auto& bound1 = bounds[id1];
        for (uint32_t j=i+1; j < groupSize; ++j) {
            float dist = distance(pointsCopy[i], pointsCopy[j]);
            auto id2 = group[j];
            auto& knn2 = idToKnn[id2];
            auto& bound2 = bounds[id2];
            if (dist < bound1) {
                knn1.addCandidateLessThan(bound1, id2, dist);
            }
            if (dist < bound2) {
                knn2.addCandidateLessThan(bound2, id1, dist);
            }
        }
    }
}


template<class TKnnSet>
void addCandidatesGroup(float points[][112],
                        vector<uint32_t>& group,
                        vector<TKnnSet>& idToKnn) {

    uint32_t groupSize = group.size();
    vector<float[112]> pointsCopy(groupSize);
    for (uint32_t i=0; i < groupSize; ++i) {
        std::memcpy(pointsCopy[i], points[group[i]], 100 * sizeof(float));
    }
    for (uint32_t i = 0; i < groupSize-1; ++i) {
        auto id1 = group[i];
        auto& knn1 = idToKnn[id1];
        for (uint32_t j=i+1; j < groupSize; ++j) {
            auto id2 = group[j];
            float dist = distance(pointsCopy[i], pointsCopy[j]);
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



static bool contains(vector<uint32_t>& currIds, uint32_t candidateId) {
    __m256i pattern = _mm256_set1_epi32(static_cast<int>(candidateId));
    auto* ids = currIds.data();

    auto limit = ids + currIds.size();
    while (ids + 8 < limit) {
        auto block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids));
        auto match = _mm256_movemask_epi8(_mm256_cmpeq_epi32(pattern, block));
        if (match) { return true; }
        ids += 8;
    }
    while (ids < limit) {
        if (*ids == candidateId) { return true; }
        ids++;
    }
    return false;
}



#endif //SIGMOD23ANN_KNNSETS_HPP
