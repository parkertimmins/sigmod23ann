#ifndef SIGMOD23ANN_SCRATCH_HPP
#define SIGMOD23ANN_SCRATCH_HPP

#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <chrono>
#include <thread>
#include <optional>
#include <mutex>
#include <limits>
#include <cstring>
#include "io.h"
#include <emmintrin.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::seconds;
using hclock = std::chrono::high_resolution_clock;
using std::unordered_map;
using std::make_pair;
using std::pair;
using std::vector;
using Range = pair<uint32_t, uint32_t>;



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

Vec scalarMult128(float c, const Vec& vec) {
    uint32_t dim = 100;
    Vec result(dim);
    __m128 cs = _mm_set1_ps(c);
    auto* v = const_cast<float*>(vec.data());
    auto* res = const_cast<float*>(result.data());
    for (uint32_t i = 0; i < dim; i+=4) {
        __m128 vs = _mm_load_ps(v);
        __m128 prod = _mm_mul_ps(cs, vs);
        _mm_store_ps(res, prod);
        vs += 4;
        res += 4;
    }
    return result;
}


float dot(const Vec &lhs, const Vec &rhs) {
    __m128 sum  = _mm_set1_ps(0);
    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    for (uint32_t i = 0; i < 100; i+=4) {
        __m128 rs = _mm_load_ps(r);
        __m128 ls = _mm_load_ps(l);
        __m128 prod = _mm_mul_ps(rs, ls);
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




float dot1(const Vec& lhs, const Vec& rhs) {
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
uint64_t alpha = 10;
uint64_t makeSignature(const Vec& randUnit, const Vec& vec) {
    return dot(randUnit, vec) / alpha;
}

unordered_map<uint64_t, float> distCache;
float distanceCached(const vector<Vec> &points, uint32_t id1, uint32_t id2) {
    uint64_t pair = id1 < id2 ? ((static_cast<uint64_t>(id1) << 32) | id2) : ((static_cast<uint64_t>(id2) << 32) | id1);
    auto iter = distCache.find(pair);
    if (iter == distCache.end()) {
        float dist = distance128(points[id1], points[id2]);
//        distCache[pair] = dist;
        distCache.insert(make_pair(pair, dist));
        return dist;
    } else {
        return iter->second;
    }
}


vector<uint32_t> CalculateOneKnn(const vector<Vec> &data,
                                 const vector<uint32_t> &sample_indexes,
                                 const uint32_t id) {
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;
    float lower_bound = std::numeric_limits<float>::max();
    for (unsigned int sample_id : sample_indexes) {
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

struct Bloom {
    const static uint8_t tableSizeLog = 9; // 512
    const static uint32_t tableSize = 1 << tableSizeLog;
    uint8_t bits[tableSize];
    uint32_t bitOffsetMask = (1 << tableSizeLog) - 1;
    uint32_t bitOffsetInByteMask = ((1<<3)-1);
    uint32_t seed = 1234567;
    uint32_t numElements = 0;
    const uint32_t maxElements = 400;

    bool mightContainSetOnFalse(uint32_t item) {
        auto h = hash(item);
        auto bitIdx = h & bitOffsetMask;
        auto byteIdx = bitIdx >> 3;
        auto bitOffsetInByte = bitIdx & bitOffsetInByteMask;

        if (bits[byteIdx] & (1 << bitOffsetInByte)) {
            return true;
        }
        numElements++;
        bits[byteIdx] |= (1 << bitOffsetInByte);
        return false;
    }

    bool mightContain(uint32_t item) {
        auto h = hash(item);
        auto bitIdx = h & bitOffsetMask;
        auto byteIdx = bitIdx >> 3;
        auto bitOffsetInByte = bitIdx & bitOffsetInByteMask;
        return bits[byteIdx] & (1 << bitOffsetInByte);
    }

    void set(uint32_t item) {
        numElements++;

        auto h = hash(item);
        auto bitIdx = h & bitOffsetMask;
        auto byteIdx = bitIdx >> 3;
        auto bitOffsetInByte = bitIdx & bitOffsetInByteMask;
        bits[byteIdx] |= (1 << bitOffsetInByte);
    }

    [[nodiscard]] uint32_t hash(uint32_t item) const {
        return __builtin_ia32_crc32si(seed, item);
    }

    void clear() {
        numElements = 0;
        std::memset(bits, 0, tableSize * sizeof(uint8_t));
    }
};


struct KnnSet {
private:
    vector<pair<float, uint32_t>> queue;
    uint32_t size = 0;
    float lower_bound = std::numeric_limits<float>::max();
public:
    KnnSet() {
        queue.resize(101);
    }

    bool contains(uint32_t node) {
        for (uint32_t i = 0; i < size; ++i) {
            auto id = queue[i].second;
            if (id == node) {
                return true;
            }
        }
        return false;
    }

    pair<float, uint32_t>& top() {
        return queue[0];
    }

    void push(pair<float, uint32_t> nodePair) {
        queue[size] = std::move(nodePair);
        size++;
        std::push_heap(queue.begin(), queue.begin() + size);
//
//        if (filter.numElements >= filter.maxElements) {
//            filter.clear();
//            for (uint32_t i = 0; i < size; ++i) {
//                filter.set(queue[i].second);
//            }
//        } else {
//            filter.set(nodePair.second);
//        }
    }

    void pop() {
        std::pop_heap(queue.begin(), queue.begin() + size);
        size--;
    }

    void addCandidate(const uint32_t candidate_id, float dist) {
        if (size < 100 && !contains(candidate_id)) {
            push(std::make_pair(dist, candidate_id));
            lower_bound = top().first;
        } else if (dist < lower_bound && !contains(candidate_id)) {
            push(std::make_pair(dist, candidate_id));
            pop();
            lower_bound = top().first;
        }
    }

    vector<uint32_t> finalize() {
        vector<uint32_t> knn;
        while (size) {
            knn.emplace_back(top().second);
            pop();
        }
        std::reverse(knn.begin(), knn.end());
        return knn;
    }
};


void addCandidatesComputeDist(const vector<Vec> &points,
                             vector<uint32_t>& indices,
                             Range range,
                             vector<KnnSetScannable>& idToKnn) {
    for (uint32_t i=range.first; i < range.second; ++i) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];
        auto& pt1 = points[id1];
        for (uint32_t j=range.first; j < range.second; ++j) {
            if (i == j) continue;
            auto id2 = indices[j];
            float dist = distance128(pt1, points[id2]);
            knn1.addCandidate(id2, dist);
        }
    }
}


void addCandidatesStoreDist(const vector<Vec> &points,
                   vector<uint32_t>& indices,
                   Range range,
                   vector<KnnSetScannable>& idToKnn,
                   vector<vector<float>>& distances) {
    auto rangeSize = range.second - range.first;

    distances.resize(rangeSize);
    for (auto& v: distances) {
        v.resize(rangeSize);
    }

    for (uint32_t i=range.first, k=0; i < range.second-1; ++i, ++k) {
        for (uint32_t j=i+1, l=k+1; j < range.second; ++j, ++l) {
            float dist = distance128(points[indices[i]], points[indices[j]]);
            distances[k][l] = dist;
            distances[l][k] = dist;
        }
    }

    for (uint32_t i=range.first, k=0; i < range.second; ++i, ++k) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];
        auto& dists = distances[k];
        for (uint32_t j=range.first, l=0; j < range.second; ++j, ++l) {
            if (k == l) continue;
            float dist = dists[l];
            auto id2 = indices[j];
            knn1.addCandidate(id2, dist);
        }
    }
}

void addCandidatesSortMerge(const vector<Vec> &points,
                            vector<uint32_t>& indices,
                            Range range,
                            vector<KnnSetScannable>& idToKnn) {

    auto rangeSize = range.second - range.first;
    vector<vector<float>> distances(rangeSize);
    for (auto& v: distances) {
        v.resize(rangeSize);
    }
    for (uint32_t i=range.first, k=0; i < range.second-1; ++i, ++k) {
        for (uint32_t j=i+1, l=k+1; j < range.second; ++j, ++l) {
            float dist = distance128(points[indices[i]], points[indices[j]]);
            distances[k][l] = dist;
            distances[l][k] = dist;
        }
    }

    auto groupSize = range.second - range.first;
    vector<pair<float, uint32_t>> distPairCache(groupSize-1);
    vector<pair<float, uint32_t>> queueScratch(100);

    for (uint32_t i=range.first, k=0; i < range.second; ++i, ++k) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];

        uint32_t ll = 0;
        for (uint32_t j=range.first, l=0; j < range.second; ++j, ++l) {
            if (k == l) continue;
            auto id2 = indices[j];
            float dist = distances[k][l];
            distPairCache[ll++] = {dist, id2};
        }

        std::sort(distPairCache.begin(), distPairCache.end());
        knn1.mergeCandidates(distPairCache, queueScratch);
    }
}

void constructResultActual(const vector<Vec>& data, vector<vector<uint32_t>>& result) {

    result.resize(data.size());

    vector<uint32_t> sample_indexes(data.size());
    iota(sample_indexes.begin(), sample_indexes.end(), 0);

    auto numThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads;

    Task<uint32_t> tasks(sample_indexes);

    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            std::optional<uint32_t> id = tasks.getTask();

            while (id) {
                result[*id] = CalculateOneKnn(data, sample_indexes, *id);

                if (*id % 100 == 0) {
                    std::cout << "completed: " << *id << std::endl;
                }
                id = tasks.getTask();
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }

    vector<uint32_t> sizes(101);
    for (uint32_t i=0; i < data.size(); ++i) {
        sizes[result[i].size()]++;
    }
    for (uint32_t i=0; i < sizes.size(); ++i) {
        std::cout << "size: " << i << ", count: " << sizes[i] << std::endl;
    }
}




// projId -> tupleId -> projection value
vector<vector<float>> projections;
// projId -> tupleId -> bucket
vector<vector<uint8_t>> buckets;
// projId -> bucketId -> [tuples]
vector<vector<vector<uint32_t>>> bucketToTuples;
// tupleId -> projId -> bucketId
 vector<vector<uint8_t>> pointToBucket;
// tupleId -> signature
vector<uint64_t> pointToSig;
// projId -> (min, max)
vector<float> projMins;
vector<float> projMaxs;

void constructResult(const vector<Vec>& points, vector<vector<uint32_t>>& result) {
    uint32_t numPoints = points.size();


    vector<Vec> projVecs = buildProjectionVecs(numProjections);

    projMins.resize(numProjections, std::numeric_limits<float>::max());
    projMaxs.resize(numProjections, std::numeric_limits<float>::min());
    projections.resize(numProjections);
    buckets.resize(numProjections);
    bucketToTuples.resize(numProjections);
    for (uint32_t projId = 0; projId < numProjections; ++projId) {
        bucketToTuples[projId].resize(numBuckets);
    }
    pointToBucket.resize(numPoints);
    for (uint32_t p = 0; p < numPoints; ++p) {
        pointToBucket[p].resize(numProjections);
    }
    pointToSig.resize(numPoints);


    for (uint32_t projId = 0; projId < numProjections; ++projId) {
        for (uint32_t ptId = 0; ptId < numPoints; ++ptId) {
            float proj = dot(projVecs[projId], points[ptId]);
            if (proj < projMins[projId]) projMins[projId] = proj;
            if (proj > projMaxs[projId]) projMaxs[projId] = proj;
            projections[projId].push_back(proj);
        }
    }

    for (uint32_t ptId = 0; ptId < numPoints; ++ptId) {
        uint64_t sig = 0;
        for (uint32_t projId = 0; projId < numProjections; ++projId) {
            float width = projMaxs[projId] - projMins[projId];
            float minProj = projMins[projId];
            float proj = projections[projId][ptId];
            float percRange = (proj - minProj) / width;
            uint8_t bucket = percRange * (numBuckets - 1);
//            std::cout << proj << " " << width << " " << percRange << " "  << static_cast<uint64_t>(bucket) << std::endl;
            sig <<= 2; // log2(maxBuckets)
            sig |= bucket;

            pointToBucket[ptId][projId] = bucket;
            buckets[projId].push_back(bucket);
            bucketToTuples[projId][bucket].push_back(ptId);
        }
        pointToSig[ptId] = sig;
    }


    std::unordered_map<uint64_t, vector<uint32_t>> sigToPoints;
    for (uint32_t ptId = 0; ptId < numPoints; ++ptId) {
        auto sig = pointToSig[ptId];
        if (sigToPoints.find(sig) == sigToPoints.end()) {
            sigToPoints[sig] = { ptId };
        } else {
            sigToPoints[sig].push_back(ptId);
        }
    }


    std::cout << "num groups: " << sigToPoints.size() << std::endl;

    result.resize(points.size());

    std::atomic<uint64_t> count = 0;
    auto startTime = hclock::now();
    auto start10k = hclock::now();

    auto keys = getKeys(sigToPoints);
    Task<uint64_t> tasks(keys);

    auto numThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads;

    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            std::optional<uint64_t> sig = tasks.getTask();

            while (sig) {
                const auto& group = sigToPoints[*sig];
                std::cout << "group size: : " << group.size() << std::endl;
                for (auto& id : group) {
                    result[id] = CalculateOneKnn(points, group, id);

                    auto localCount = count++;
                    if (localCount % 10'000 == 0) {
                        auto currentTime = hclock::now();
                        auto durationGroup = duration_cast<milliseconds>(currentTime - start10k);
                        auto durationTotal = duration_cast<milliseconds>(currentTime - startTime);

                        auto percentDone = static_cast<float>(localCount) / points.size();
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
    for (uint32_t i=0; i < points.size(); ++i) {
        sizes[result[i].size()]++;
        while (result[i].size() < 100)  {
            result[i].push_back(unusedId);
        }
    }

    for (uint32_t i=0; i < sizes.size(); ++i) {
        std::cout << "size: " << i << ", count: " << sizes[i] << std::endl;
    }
}


vector<Vec> buildProjectionVecs(auto n) {
    vector<Vec> randVecs;
    while (randVecs.size() < n) {
        randVecs.push_back(randUniformUnitVec(100));
    }
    return randVecs;
}


Vec pca1(vector<pair<float, uint32_t>>& group, const vector<Vec>& vecs) {
    auto dim = vecs[0].size();
    auto means = getMeans(group, vecs);

    auto r = randUniformUnitVec();
    for (auto c = 0; c < 10; ++c) {
        Vec s(dim, 0);
        for (auto& [hash, id] : group) {
            auto& v = vecs[id];
            auto x = sub(v, means);
            plusEq(s, scalarMult(dot(x, r), x));
        }
//        auto lambda = dot(r, s);

        normalizeInPlace(s);
        r = s;

//        std::cout << "pca, iteration: " << c << std::endl;
//        std::cout << "error: "; print(error);
//        std::cout << "error norm: " << norm(error);
//        std::cout << "r: "; print(r);
//        std::cout << std::endl;
    }
    return r;
}


Vec pca1(const vector<Vec>& vecs) {
    auto dim = vecs[0].size();
    auto means = getMeans(vecs);

    auto r = randUniformUnitVec();

    for (auto c = 0; c < 10; ++c) {
        Vec s(dim, 0);
        for (auto& v : vecs) {
            auto x = sub(v, means);
            plusEq(s, scalarMult(dot256(x, r), x));
        }
//        auto lambda = dot(r, s);

        normalizeInPlace(s);
        r = s;

//        std::cout << "elapsed: " << duration_cast<milliseconds>(hclock::now() - start).count() << std::endl;
//        std::cout << "pca, iteration: " << c << std::endl;
//        std::cout << "error norm: " << norm(error) << std::endl;
//        std::cout << "error: "; print(error);
//        std::cout << "r: "; print(r);
//        std::cout << std::endl;
    }
    return r;
}


void splitNoSortMulti(const vector<Vec>& points,vector<pair<float, uint32_t>>& group1, vector<vector<pair<float, uint32_t>>>& allGroups) {
    vector<vector<pair<float, uint32_t>>> stack;
    stack.push_back(group1);
    std::mutex stack_mtx;
    std::mutex allGroup_mtx;

    auto numPoints = points.size();
    auto numThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads;
    std::atomic<uint32_t> count = 0;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            while (count < numPoints) {
                stack_mtx.lock();
                if (!stack.empty()) {
                    auto group = stack.back(); stack.pop_back();
                    stack_mtx.unlock();

                    uint32_t numSplits = 4;
                    if (group.size() < maxGroupSize) {
                        count += group.size();
                        std::lock_guard<std::mutex> guard(allGroup_mtx);
                        allGroups.push_back(group);
                    } else {
                        // modify group in place
                        auto [min, max] = rehashMinMax(group, points);
                        auto splitSize = (max - min) / numSplits;

                        vector<vector<pair<float, uint32_t>>> splits(numSplits);
                        for (auto& p : group) {
                            auto& [hash, id] = p;
                            uint32_t splitIdx = (hash - min) / splitSize;
                            splitIdx = std::min(splitIdx, numSplits - 1);
                            splits[splitIdx].push_back(p);
                        }
                        {
                            std::lock_guard<std::mutex> guard(stack_mtx);
                            stack.insert(stack.end(), splits.begin(), splits.end());
                        }
                    }
                } else {
                    stack_mtx.unlock();
                }
            }

        });
    }

    for (auto& thread: threads) { thread.join(); }
}



vector<pair<float, uint32_t>> buildInitialGroups(const vector<Vec>& points) {
    vector<pair<float, uint32_t>> group;
    auto numPoints = points.size();
    auto u = randUniformUnitVec();
    for (uint32_t i = 0; i < numPoints; ++i) {
        float hash = dot(u, points[i]);
        group.emplace_back(hash, i);
    }
    sort(group.begin(), group.end());
    return group;
}


Vec sub128(const Vec& lhs, const Vec& rhs) {
    auto dim = lhs.size();
    Vec result(dim);
    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    auto* res = const_cast<float*>(result.data());
    for (uint32_t i = 0; i < dim; i+=4) {
        __m128 ls = _mm_load_ps(l);
        __m128 rs = _mm_load_ps(r);
        __m128 diff = _mm_sub_ps(ls, rs);
        _mm_store_ps(res, diff);
        r += 4;
        l += 4;
        res += 4;
    }
    return result;
}



void plusEq(Vec& lhs, const Vec& rhs) {
    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    for (uint32_t i = 0; i < 100; i+=4) {
        __m128 ls = _mm_load_ps(l);
        __m128 rs = _mm_load_ps(r);
        __m128 sum = _mm_add_ps(ls, rs);
        _mm_store_ps(l, sum);
        l += 4;
        r += 4;
    }
}

template<class K, class V>
vector<K> getKeys(std::unordered_map<K, V>& map) {
    vector<K> keys;
    for(const auto& kv : map) {
        keys.push_back(kv.first);
    }
    return keys;
}

void print(const vector<float>& ts) {
    for (float t: ts)
        std::cout << t << ", ";
    std::cout << std::endl;
}

Vec getMeans(const vector<Vec>& vecs) {
    uint32_t dim = 100;
    uint32_t numRows = vecs.size();
    vector<double> sums(dim, 0);
    for (auto& v : vecs) {
        for (uint32_t i = 0; i < dim; ++i) {
            sums[i] += v[i];
        }
    }
    Vec means;
    for (auto& sum : sums) {
        means.push_back(sum / numRows);
    }
    return means;
}



Vec getMeans(vector<pair<float, uint32_t>>& group, const vector<Vec>& vecs) {
    uint32_t dim = 100;
    uint32_t numRows = group.size();
    vector<double> sums(dim, 0);

    for (auto& [hash, id] : group) {
        auto& v = vecs[id];
        for (uint32_t i = 0; i < dim; ++i) {
            sums[i] += v[i];
        }
    }
    Vec means;
    for (auto& sum : sums) {
        means.push_back(sum / numRows);
    }
    return means;
}


void splitNoSortHalf(const vector<Vec>& points,vector<pair<float, uint32_t>>& group1, vector<vector<pair<float, uint32_t>>>& allGroups) {
    vector<vector<pair<float, uint32_t>>> stack;
    stack.push_back(group1);
    std::mutex stack_mtx;
    std::mutex allGroup_mtx;

    auto numPoints = points.size();
    auto numThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads;
    std::atomic<uint32_t> count = 0;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            while (count < numPoints) {
                stack_mtx.lock();
                if (!stack.empty()) {
                    auto group = stack.back(); stack.pop_back();
                    stack_mtx.unlock();

                    if (group.size() < maxGroupSize) {
                        count += group.size();
                        std::lock_guard<std::mutex> guard(allGroup_mtx);
                        allGroups.push_back(group);
                    } else {
                        // modify group in place
                        auto [min, max] = rehashMinMax(group, points);
                        auto mid = min + (max - min) / 2;

                        vector<pair<float, uint32_t>> low;
                        vector<pair<float, uint32_t>> hi;
                        for (auto& p : group) {
                            auto& [hash, id] = p;
                            if (hash <= mid) {
                                low.push_back(p);
                            } else {
                                hi.push_back(p);
                            }
                        }
                        {
                            std::lock_guard<std::mutex> guard(stack_mtx);
                            stack.push_back(low);
                            stack.push_back(hi);
                        }
                    }
                } else {
                    stack_mtx.unlock();
                }
            }

        });
    }

    for (auto& thread: threads) { thread.join(); }
}


void splitRecursiveMulti(const vector<Vec>& points,vector<pair<float, uint32_t>>& group, vector<vector<pair<float, uint32_t>>>& allGroups) {
    if (group.size() < maxGroupSize) {
        allGroups.push_back(group);
    } else {
        // modify group in place
        rehash(group, points);
        sort(group.begin(), group.end());

        auto numSplits = 4;
        uint32_t splitSize = group.size() / numSplits;
        for (auto split = 0; split < numSplits - 1; ++split) {
            vector<pair<float, uint32_t>> splitGroup;
            while (splitGroup.size() < splitSize) {
                splitGroup.push_back(group.back());
                group.pop_back();
            }
            splitRecursiveMulti(points,  splitGroup, allGroups);
        }
        splitRecursiveMulti(points, group, allGroups);
    }
}



void splitRecursiveNoSort(const vector<Vec>& points,vector<pair<float, uint32_t>>& group, vector<vector<pair<float, uint32_t>>>& allGroups) {
    if (group.size() < maxGroupSize) {
        allGroups.push_back(group);
    } else {
        // modify group in place
        auto [min, max] = rehashMinMax(group, points);
        auto mid = min + (max - min) / 2;

        vector<pair<float, uint32_t>> low;
        vector<pair<float, uint32_t>> hi;
        for (auto& p : group) {
            auto& [hash, id] = p;
            if (hash <= mid) {
                low.push_back(p);
            } else {
                hi.push_back(p);
            }
        }
        splitRecursiveNoSort(points, low, allGroups);
        splitRecursiveNoSort(points, hi, allGroups);
    }
}




#endif //SIGMOD23ANN_SCRATCH_HPP
