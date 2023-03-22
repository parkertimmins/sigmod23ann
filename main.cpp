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
#include <boost/align/aligned_alloc.hpp>

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
pair<float, float> startBounds = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};



/**
 * Docs used
 * https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
 * https://en.wikipedia.org/wiki/Locality-sensitive_hashing
 * http://infolab.stanford.edu/~ullman/mining/2009/similarity3.pdf
 * http://infolab.stanford.edu/~ullman/mining/pdf/cs345-lsh.pdf
 * http://infolab.stanford.edu/~ullman/mining/2008/slides/cs345-lsh.pdf
 * https://users.cs.utah.edu/~jeffp/teaching/cs5955/L6-LSH.pdf
 * http://infolab.stanford.edu/~ullman/mmds/ch3.pdf
 * http://web.mit.edu/andoni/www/papers/cSquared.pdf
 * https://courses.engr.illinois.edu/cs498abd/fa2020/slides/14-lec.pdf
 * https://www.youtube.com/watch?v=yIkyeackISs&ab_channel=SimonsInstitute
 * https://arxiv.org/abs/1501.01062
 * http://www.slaney.org/malcolm/yahoo/Slaney2012(OptimalLSH).pdf
 * https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf
 * https://people.csail.mit.edu/indyk/icm18.pdf
 * https://arxiv.org/pdf/1806.09823.pdf - Approximate Nearest Neighbor Search in High Dimensions
 * https://people.csail.mit.edu/indyk/p117-andoni.pdf
 * https://www.youtube.com/watch?v=cn15P8vgB1A&ab_channel=RioICM2018
 */


uint64_t groupingTime = 0;
uint64_t processGroupsTime = 0;
std::default_random_engine rd(123);
const uint32_t dims = 100;
const uint32_t k = 100;


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


float distance128(const float* lhs, const float* rhs) {
    __m128 sum  = _mm_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < dims; i+=4) {
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


float distance(const float* lhs, const float* rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        __m256 diff = _mm256_sub_ps(ls, rs);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        r += 8;
        l += 8;
    }

    alignas(sizeof(__m256)) float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < dims; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}

double norm(const Vec& vec) {
    float sumSquares = 0.0;
    for (auto& v : vec) {
        sumSquares += (v * v);
    }
    return sqrt(sumSquares);
}

void normalizeInPlace(Vec& vec) {
    double vecNorm = norm(vec);
    for (float& v : vec) {
        v = v / vecNorm;
    }
}

Vec normalize(const Vec& vec) {
    Vec res;
    double vecNorm = norm(vec);
    for (float v : vec) {
        res.push_back(v / vecNorm);
    }
    return vec;
}

Vec randUniformUnitVec(size_t dim=dims) {
    std::mt19937 gen(rd());
    std::normal_distribution<> normalDist(0, 1);

    Vec randVec;
    randVec.reserve(dim);
    while (randVec.size() < dim) {
        auto elementValue = normalDist(gen);
        randVec.emplace_back(elementValue);
    }

    // make unit vector just for good measure
    normalizeInPlace(randVec);
    return randVec;
}

Vec sub(const Vec& lhs, const Vec& rhs) {
    auto dim = lhs.size();
    Vec result(dim);
    for (uint32_t i = 0; i < dim; i++) {
        result[i]  = lhs[i] - rhs[i];
    }
    return result;
}

Vec scalarMult(float c, const Vec& vec) {
    Vec res;
    for (float v : vec) {
        res.push_back(c * v);
    }
    return res;
}



float dot128(const float* lhs, const float* rhs) {
    __m128 sum  = _mm_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < dims; i+=4) {
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



float dot(const float* lhs, const float* rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        sum = _mm256_fmadd_ps(rs, ls, sum);
        l += 8;
        r += 8;
    }
    alignas(sizeof(__m256)) float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < dims; ++i) {
        ans += (lhs[i] * rhs[i]);
    }
    return ans;
}


// project v onto u
Vec project(const Vec& u, const Vec& v) {
    return scalarMult(dot(u.data(), v.data()), normalize(u));
}


struct KnnSetScannable {
private:
    vector<pair<float, uint32_t>> queue;
    uint32_t size = 0;
    float lower_bound = std::numeric_limits<float>::max();
public:
    KnnSetScannable() {
        queue.resize(k);
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

    void append(pair<float, uint32_t> nodePair) {
        queue[size] = std::move(nodePair);
        size++;
    }

    // This may misorder nodes of equal sizes
    void addCandidate(const uint32_t candidate_id, float dist) {
        if (size < k && !contains(candidate_id)) {
            append({dist, candidate_id});
            lower_bound = std::max(lower_bound, dist);
        } else if (dist < lower_bound) {
            float secondMaxVal = std::numeric_limits<float>::min();
            float maxVal = std::numeric_limits<float>::min();
            uint32_t maxIdx = -1;
            for (uint32_t i = 0; i < size; ++i) {
                auto& [otherDist, id] = queue[i];
                if (id == candidate_id) {
                    return;
                }

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


void addCandidates(float points[][104],
                   vector<uint32_t>& indices,
                   Range range,
                   vector<KnnSetScannable>& idToKnn) {
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

vector<pair<float, uint32_t>> splitInTwo(vector<pair<float, uint32_t>>& group1) {
    sort(group1.begin(), group1.end());
    vector<pair<float, uint32_t>> group2;
    uint32_t half = group1.size() / 2;
    while (group2.size() < half) {
        group2.push_back(group1.back());
        group1.pop_back();
    }
    return group2;
}

// assume that argument unit vectors are linearly independent
// also unit vectors
vector<Vec> gramSchmidt(vector<Vec>& v) {
    vector<Vec> u(v.size());
    u[0] = normalize(v[0]);
    for (uint32_t i = 1; i < v.size(); ++i) {
        u[i] = v[i];
        for (uint32_t j = 0; j < i; ++j) {
            u[i] = sub(u[i], project(u[j], v[i]));
        }
        u[i] = normalize(u[i]);
    }
    return u;
}

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

pair<float, float> getBounds(vector<vector<float>>& hashes, std::vector<uint32_t>& indices, Range range, uint32_t depth) {
    auto [min, max] = startBounds;
    for (uint32_t i = range.first; i < range.second; ++i) {
        auto id = indices[i];
        auto proj = hashes[id][depth];
        min = std::min(min, proj);
        max = std::max(max, proj);
    }
    return {min, max};
}

uint32_t requiredHashFuncs(uint32_t numPoints, uint32_t maxBucketSize) {
    uint32_t groupSize = numPoints;
    uint32_t numHashFuncs = 0;
    while (groupSize > maxBucketSize) {
        groupSize /= 2;
        numHashFuncs++;
    }
    return numHashFuncs;
}

// assume range size is at least numIndices
float getSplitPoint(vector<vector<float>>& hashes, std::vector<uint32_t>& indices, Range range, uint32_t depth) {
    auto rangeSize = range.second - range.first;

    if (rangeSize < 5'000) {
        auto [min, max] = getBounds(hashes, indices, range, depth);
        return (min + max) / 2;
    }

    uint32_t numIndices = pow(log(rangeSize), 1.5);
    std::unordered_set<uint32_t> randIndices;
    std::uniform_int_distribution<> distribution(range.first, range.second - 1);
    while (randIndices.size() < numIndices) {
        uint32_t idx = distribution(rd);
        randIndices.insert(idx);
    }

    auto [min, max] = startBounds;
    for (auto idx : randIndices) {
        auto id = indices[idx];
        auto proj = hashes[id][depth];
        min = std::min(min, proj);
        max = std::max(max, proj);
    }
    return (min + max) / 2;
}


void splitHorizontalHistogram(uint32_t numHashFuncs, uint32_t numPoints, float points[][104], std::unordered_map<uint64_t, vector<uint32_t>>& globalGroups) {
    auto startHash = hclock::now();

    auto numThreads = std::thread::hardware_concurrency();
    auto hashRanges = splitRange({0, numPoints}, numThreads);
    vector<vector<float>> hashes(numPoints);

    vector<Vec> unitVecs(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        unitVecs[h] = randUniformUnitVec();
    }
    unitVecs = gramSchmidt(unitVecs);


    
    vector<vector<pair<float, float>>> localBounds(numThreads);
    
    // compute all hash function values
    vector<std::thread> threads;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto& bounds = localBounds[t];
            bounds.resize(numHashFuncs, startBounds);
            auto range = hashRanges[t];
            for (uint32_t i = range.first; i < range.second; ++i) {
                const float* p = points[i];
                vector<float>& hashSet = hashes[i];
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    const auto& unitVec = unitVecs[h];
                    float proj = dot(unitVec.data(), p);
                    hashSet.push_back(proj);
                    auto& [min, max] = bounds[h];
                    min = std::min(min, proj);
                    max = std::max(max, proj);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }


    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << std::endl;
    auto startSplit = hclock::now();

    // merge threadlocal bounds
    vector<pair<float, float>> globalBounds(numHashFuncs, startBounds);
    for (auto& bounds : localBounds) {
        for (uint32_t h = 0; h < numHashFuncs; ++h) {
            auto& [minLocal, maxLocal] = bounds[h];
            auto& [minGlobal, maxGlobal] = globalBounds[h];
            minGlobal = std::min(minGlobal, minLocal);
            maxGlobal = std::max(maxGlobal, maxLocal);
        }
    }

    // compute local histograms
    const uint32_t numHistogramBuckets = 100;
    threads.clear();
    vector<vector<vector<uint32_t>>> localHistograms(numThreads);
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto& histos = localHistograms[t];
            histos.resize(numHashFuncs);
            for (uint32_t h = 0; h < numHashFuncs; ++h) {
                histos[h].resize(numHistogramBuckets, 0);
            }

            auto range = hashRanges[t];
            for (uint32_t i = range.first; i < range.second; ++i) {
                vector<float>& hashSet = hashes[i];
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    float proj = hashSet[h];
                    auto& [min, max] = globalBounds[h];
                    float width = max - min;
                    float bucketWidth = width / numHistogramBuckets;
                    auto bucket = std::min(static_cast<uint32_t>((proj - min) / bucketWidth), numHistogramBuckets - 1);
                    histos[h][bucket] +=1;
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    // merge local histograms
    vector<vector<uint32_t>> globalHistograms(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        globalHistograms[h].resize(numHistogramBuckets, 0);
    }
    for (auto& histos: localHistograms) {
        for (uint32_t h = 0; h < numHashFuncs; ++h) {
            for (uint32_t b = 0; b < numHistogramBuckets; ++b) {
                globalHistograms[h][b] += histos[h][b];
            }
        }
    }

    // compute median split points
    vector<float> medianSplits(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        uint32_t pointsSeen = 0;
        for (uint32_t b = 0; b < numHistogramBuckets; ++b) {
            pointsSeen += globalHistograms[h][b];
            // bucket contains median
            if (pointsSeen >= numPoints / 2) {
                auto& [min, max] = globalBounds[h];
                float width = max - min;
                float bucketWidth = width / numHistogramBuckets;

                float bucketStart = b * bucketWidth + min;
                float bucketEnd = bucketStart + bucketWidth;
                std::uniform_real_distribution<float> distribution(bucketStart, bucketEnd);
                // sample from bucket containing median
                float median = distribution(rd);
                medianSplits[h] = median;
                break;
            }
        }
    }

    // aggregate based on splits into local maps
    threads.clear();
    vector<std::unordered_map<uint64_t, vector<uint32_t>>> localGroups(numThreads);
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto& groups = localGroups[t];
            auto range = hashRanges[t];
            for (uint32_t i = range.first; i < range.second; ++i) {
                vector<float>& hashSet = hashes[i];
                uint64_t sig = 0;
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    float proj = hashSet[h];
                    float median = medianSplits[h];
                    if (proj > median) {
                        sig |= (1 << h);
                    }
                }

                if (groups.find(sig) == groups.end()) {
                    groups[sig] = vector<uint32_t>();
                }
                groups[sig].push_back(i);
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }


    for (unordered_map<uint64_t, vector<uint32_t>>& localGroupSet: localGroups) {
        for (auto& [sig, group] : localGroupSet) {
            if (globalGroups.find(sig) == globalGroups.end()) {
                globalGroups[sig] = vector<uint32_t>();
            }
            auto& globalGroup = globalGroups[sig];
            globalGroup.insert(globalGroup.end(), group.begin(), group.end());
        }
    }
    std::cout << "histogram split time: " << duration_cast<milliseconds>(hclock::now() - startSplit).count() << std::endl;
}



void splitHorizontalThreadArray(uint32_t maxGroupSize, uint32_t numHashFuncs, uint32_t numPoints, float points[][104], vector<Range>& ranges, vector<uint32_t>& indices) {
    auto numThreads = std::thread::hardware_concurrency();
    auto hashRanges = splitRange({0, numPoints}, numThreads);
    vector<vector<float>> hashes(numPoints);

    vector<Vec> unitVecs(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        unitVecs[h] = randUniformUnitVec();
    }
    unitVecs = gramSchmidt(unitVecs);

    auto startHash = hclock::now();

    // compute all hash function values
    vector<std::thread> threads;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto range = hashRanges[t];
            for (uint32_t i = range.first; i < range.second; ++i) {
                const float* p = points[i];
                vector<float>& hashSet = hashes[i];
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    const auto& unitVec = unitVecs[h];
                    float proj = dot(unitVec.data(), p);
                    hashSet.push_back(proj);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << std::endl;

    auto startRegroup = hclock::now();

    vector<pair<uint32_t, Range>> stack;
    stack.emplace_back(0, make_pair(0, numPoints));
    std::mutex stack_mtx;
    std::mutex groups_mtx;
    uint32_t actualMaxGroupsSize = 0;

    threads.clear();
    std::atomic<uint32_t> count = 0;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            while (count < numPoints) {
                stack_mtx.lock();
                if (!stack.empty()) {
                    auto [depth, range] = stack.back(); stack.pop_back();
                    stack_mtx.unlock();
                    uint32_t rangeSize = range.second - range.first;

                    if (rangeSize < maxGroupSize || depth == numHashFuncs) {
                        count += rangeSize;
                        std::lock_guard<std::mutex> guard(groups_mtx);
                        ranges.push_back(range);
                        actualMaxGroupsSize = std::max(rangeSize, actualMaxGroupsSize);
                    } else {
                        auto [min, max] = getBounds(hashes, indices, range, depth);
                        auto mid = (min + max) / 2;
                        auto rangeBegin = indices.begin() + range.first;
                        auto rangeEnd = indices.begin() + range.second;
                        auto middleIt = std::partition(rangeBegin, rangeEnd, [&](uint32_t id) {
                            auto proj = hashes[id][depth];
                            return proj <= mid;
                        });
                        auto range1Size = middleIt - rangeBegin;
                        Range lo = {range.first, range.first + range1Size};
                        Range hi = {range.first + range1Size , range.second};
                        {
                            std::lock_guard<std::mutex> guard(stack_mtx);
                            stack.emplace_back(depth+1, lo);
                            stack.emplace_back(depth+1, hi);
                        }
                    }
                } else {
                    stack_mtx.unlock();
                }
            }

        });
    }
    for (auto& thread: threads) { thread.join(); }

    std::cout << "group regroup time: " << duration_cast<milliseconds>(hclock::now() - startRegroup).count() << std::endl;
    std::cout << "group regroup maximum group size: " << actualMaxGroupsSize << std::endl;
}



void splitHorizontalThreadVector(uint32_t maxGroupSize, uint32_t numHashFuncs, const vector<Vec>& points, vector<Range>& ranges, vector<uint32_t>& indices) {
    auto numPoints = points.size();
    auto numThreads = std::thread::hardware_concurrency();
    auto hashRanges = splitRange({0, numPoints}, numThreads);
    vector<vector<float>> hashes(numPoints);

    vector<Vec> unitVecs(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        unitVecs[h] = randUniformUnitVec();
    }
    unitVecs = gramSchmidt(unitVecs);

    auto startHash = hclock::now();

    // compute all hash function values
    vector<std::thread> threads;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto range = hashRanges[t];
            for (uint32_t i = range.first; i < range.second; ++i) {
                Vec p = points[i];
                vector<float>& hashSet = hashes[i];
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    const auto& unitVec = unitVecs[h];
                    float proj = dot(unitVec.data(), p.data());
                    hashSet.push_back(proj);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << std::endl;

    auto startRegroup = hclock::now();

    vector<pair<uint32_t, Range>> stack;
    stack.emplace_back(0, make_pair(0, numPoints));
    std::mutex stack_mtx;
    std::mutex groups_mtx;

    threads.clear();
    std::atomic<uint32_t> count = 0;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            while (count < numPoints) {
                stack_mtx.lock();
                if (!stack.empty()) {
                    auto [depth, range] = stack.back(); stack.pop_back();
                    stack_mtx.unlock();
                    uint32_t rangeSize = range.second - range.first;

                    if (rangeSize < maxGroupSize || depth == numHashFuncs) {
                        count += rangeSize;
                        std::lock_guard<std::mutex> guard(groups_mtx);
                        ranges.push_back(range);
                    } else {
                        auto mid = getSplitPoint(hashes, indices, range, depth);
                        auto rangeBegin = indices.begin() + range.first;
                        auto rangeEnd = indices.begin() + range.second;
                        auto middleIt = std::partition(rangeBegin, rangeEnd, [&](uint32_t id) {
                            auto proj = hashes[id][depth];
                            return proj <= mid;
                        });
                        auto range1Size = middleIt - rangeBegin;
                        Range lo = {range.first, range.first + range1Size};
                        Range hi = {range.first + range1Size , range.second};
                        {
                            std::lock_guard<std::mutex> guard(stack_mtx);
                            stack.emplace_back(depth+1, lo);
                            stack.emplace_back(depth+1, hi);
                        }
                    }
                } else {
                    stack_mtx.unlock();
                }
            }

        });
    }
    for (auto& thread: threads) { thread.join(); }

    std::cout << "group regroup time: " << duration_cast<milliseconds>(hclock::now() - startRegroup).count() << std::endl;
}

void splitSortForAdjacency(vector<Vec>& pointsRead, std::vector<uint32_t>& newToOldIndices, float points[][104], uint32_t numThreads, uint32_t numPoints, vector<Range>& ranges) {
    auto startAdjacencySort = hclock::now();
    std::iota(newToOldIndices.begin(), newToOldIndices.end(), 0);
    splitHorizontalThreadVector(200, requiredHashFuncs(pointsRead.size(), 200), pointsRead, ranges, newToOldIndices);
    vector<Range> moveRanges = splitRange({0, numPoints}, numThreads);
    vector<std::thread> threads;
    for (uint32_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto moveRange = moveRanges[t];
            for (uint32_t newIdx = moveRange.first; newIdx < moveRange.second; ++newIdx) {
                auto oldIdx = newToOldIndices[newIdx];
                memcpy(points[newIdx], pointsRead[oldIdx].data(), sizeof(float) * dims);
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    pointsRead.clear();
    auto adjacencySortDuration = duration_cast<milliseconds>(hclock::now() - startAdjacencySort).count();
    std::cout << "adjacency sort time: " << adjacencySortDuration << std::endl;
}


template<class K, class V>
vector<K> getKeys(std::unordered_map<K, V>& map) {
    vector<K> keys;
    for(const auto& kv : map) {
        keys.push_back(kv.first);
    }
    return keys;
}


void constructResultSplitting(vector<Vec>& pointsRead, vector<vector<uint32_t>>& result) {

    long timeBoundsMs;
    if(getenv("LOCAL_RUN")) {
        timeBoundsMs = 60'000;
    } else {
        timeBoundsMs = pointsRead.size() == 10'000 ? 20'000 : 1'600'000;
    }

    std::cout << "start run with time bound: " << timeBoundsMs << std::endl;

    auto startTime = hclock::now();
    vector<KnnSetScannable> idToKnn(pointsRead.size());
    uint32_t numPoints = pointsRead.size();
    auto numThreads = std::thread::hardware_concurrency();

    // rewrite point data in adjacent memory and sort in a group order
    vector<Range> ranges;
    std::vector<uint32_t> newToOldIndices(numPoints);
    float (*points)[104] = reinterpret_cast<float(*)[104]>(new __m256[(numPoints * 104 * sizeof(float)) / sizeof(__m256)]);
    splitSortForAdjacency(pointsRead, newToOldIndices, points, numThreads, numPoints, ranges);

    uint32_t iteration = 0;
    while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
        std::cout << "Iteration: " << iteration << std::endl;


        auto startGroup = hclock::now();
        uint32_t numHashFuncs = requiredHashFuncs(numPoints, 300);
        std::unordered_map<uint64_t, vector<uint32_t>> globalGroups;
        splitHorizontalHistogram(numHashFuncs, numPoints, points, globalGroups);
        auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
        groupingTime += groupDuration;

        auto startProcessing = hclock::now();

        vector<std::thread> threads;
        std::mutex groupMtx;
        std::atomic<uint32_t> count = 0;

        auto signatures = getKeys(globalGroups);
        Task<uint64_t> tasks(signatures);
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                auto optSig = tasks.getTask();
                while (optSig) {
                    auto sig = *optSig;
                    auto& group = globalGroups[sig];
                    count += group.size();
//                        if (group.size() <= 300) {
                    addCandidatesGroup(points, group, idToKnn);
//                        }
                    optSig = tasks.getTask();
                }
            });
        }

        for (auto& thread: threads) { thread.join(); }

        auto processingDuration = duration_cast<milliseconds>(hclock::now() - startProcessing).count();
        processGroupsTime += processingDuration;

        std::cout << "processing time: " << processingDuration << std::endl;


        std::cout << "--------------------------------------------------------------------------------------------------------" << std::endl;
        iteration++;
    }

    for (uint32_t id = 0; id < numPoints; ++id) {
        auto newIdxResultRow = idToKnn[id].finalize();
        for (auto& ni : newIdxResultRow) {
            ni = newToOldIndices[ni];
        }
        result[newToOldIndices[id]] = std::move(newIdxResultRow);
    }

//    for (uint32_t id = 0; id < points.size(); ++id) {
//        result[id] = idToKnn[id].finalize();
//    }

    auto sizes = padResult(numPoints, result);
    for (uint32_t i=0; i < sizes.size(); ++i) {
        std::cout << "size: " << i << ", count: " << sizes[i] << std::endl;
    }

    std::cout << "total grouping time (ms): " << groupingTime << std::endl;
    std::cout << "total processing time (ms): " << processGroupsTime << std::endl;

}

int main(int argc, char **argv) {
  auto startTime = hclock::now();

  string source_path = "dummy-data.bin";

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  // Read data points
  vector<Vec> nodes;

  auto startRead = hclock::now();
  ReadBin(source_path, nodes);
  std::cout << "read time: " << duration_cast<milliseconds>(hclock::now() - startRead).count() << std::endl;

  // Knng constuction
  vector<vector<uint32_t>> knng(nodes.size());
  constructResultSplitting(nodes, knng);

  // Save to ouput.bin
  auto startSave = hclock::now();
  SaveKNNG(knng);
  std::cout << "save time: " << duration_cast<milliseconds>(hclock::now() - startSave).count() << std::endl;

  auto totalDuration = duration_cast<milliseconds>(hclock::now() - startTime).count();
  std::cout << "total time (ms): " << totalDuration << std::endl;
  return 0;
}

