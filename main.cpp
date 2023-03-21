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


uint64_t maxGroupSize = 200;
uint64_t groupingTime = 0;
uint64_t processGroupsTime = 0;


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

float distance256(const Vec &lhs, const Vec &rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        __m256 diff = _mm256_sub_ps(ls, rs);
        __m256 prod = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, prod);
        r += 8;
        l += 8;
    }
    float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < 100; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}


//float distance512(const Vec &lhs, const Vec &rhs) {
//    __m512 sum  = _mm512_set1_ps(0);
//    auto* r = const_cast<float*>(rhs.data());
//    auto* l = const_cast<float*>(lhs.data());
//    for (uint32_t i = 0; i < 96; i+=16) {
//        __m512 rs = _mm512_load_ps(r);
//        __m512 ls = _mm512_load_ps(l);
//        __m512 diff = _mm512_sub_ps(ls, rs);
//        __m512 prod = _mm512_mul_ps(diff, diff);
//        sum = _mm512_add_ps(sum, prod);
//        r += 16;
//        l += 16;
//    }
//    float sums[16] = {};
//    _mm512_store_ps(sums, sum);
//    float ans = 0.0f;
//    for (float s: sums) {
//        ans += s;
//    }
//    for (unsigned i = 96; i < 100; ++i) {
//        auto d = (lhs[i] - rhs[i]);
//        ans += (d * d);
//    }
//    return ans;
//}

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

std::default_random_engine rd(123);
Vec randUniformUnitVec(size_t dim=100) {
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


float dot256(const Vec &lhs, const Vec &rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = const_cast<float*>(rhs.data());
    auto* l = const_cast<float*>(lhs.data());
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        __m256 prod = _mm256_mul_ps(rs, ls);
        sum = _mm256_add_ps(sum, prod);
        l += 8;
        r += 8;
    }
    float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < 100; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}


// project v onto u
Vec project(const Vec& u, const Vec& v) {
    return scalarMult(dot256(u, v), normalize(u));
}


struct KnnSetScannable {
private:
    vector<pair<float, uint32_t>> queue;
    uint32_t size = 0;
    float lower_bound = std::numeric_limits<float>::max();
public:
    KnnSetScannable() {
        queue.resize(100);
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
        if (size < 100 && !contains(candidate_id)) {
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

        while (out < 100 && (l < left.size() || r < rightSize)) {
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
                while (out < 100 && l < left.size())  {
                    if (left[l] == output[out - 1]) {
                        l++;
                    } else {
                        output[out++] = left[l++];
                    }
                }
            } else {
                while (out < 100 && r < rightSize)  {
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


void addCandidates(const vector<Vec> &points,
                   vector<uint32_t>& indices,
                   Range range,
                   vector<KnnSetScannable>& idToKnn) {
    for (uint32_t i=range.first; i < range.second-1; ++i) {
        auto id1 = indices[i];
        auto& knn1 = idToKnn[id1];
        for (uint32_t j=i+1; j < range.second; ++j) {
            auto id2 = indices[j];
            float dist = distance256(points[id1], points[id2]);
            knn1.addCandidate(id2, dist);
            idToKnn[id2].addCandidate(id1, dist);
        }
    }
}

vector<uint32_t> padResult(const vector<Vec>& points, vector<vector<uint32_t>>& result) {
    auto unusedId = 1;
    vector<uint32_t> sizes(101);
    for (uint32_t i=0; i < points.size(); ++i) {
        sizes[result[i].size()]++;
        while (result[i].size() < 100)  {
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

void rehash(vector<pair<float, uint32_t>>& group, const vector<Vec>& vecs) {
    auto u = randUniformUnitVec();
    for (auto& p : group) {
        p.first = dot256(u, vecs[p.second]);
    }
}

pair<float, float> rehashMinMax(vector<pair<float, uint32_t>>& group, const vector<Vec>& vecs) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    auto u = randUniformUnitVec();
    for (auto& p : group) {
        float proj = dot256(u, vecs[p.second]);
        p.first = proj;
        min = std::min(proj, min);
        max = std::max(proj, max);
    }
    return {min, max};
}

void splitRecursiveSingleThreaded(const vector<Vec>& points,vector<pair<float, uint32_t>>& group, vector<vector<pair<float, uint32_t>>>& allGroups) {
    if (group.size() < maxGroupSize) {
        allGroups.push_back(group);
    } else {
        // modify group in place
        rehash(group, points);
        auto otherGroup = splitInTwo(group);
        splitRecursiveSingleThreaded(points, group, allGroups);
        splitRecursiveSingleThreaded(points, otherGroup, allGroups);
    }
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

pair<float, float> startBounds = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
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

void splitHorizontalThread(uint32_t numHashFuncs, const vector<Vec>& points, vector<Range>& ranges, vector<uint32_t>& indices) {
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
                const Vec& vec = points[i];
                vector<float>& hashSet = hashes[i];
                for (uint32_t h = 0; h < numHashFuncs; ++h) {
                    const auto& unitVec = unitVecs[h];
                    float proj = dot256(unitVec, vec);
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
                        auto [min, max] = getBounds(hashes, indices, range, depth);
                        auto mid = min + (max - min) / 2;

                        auto rangeBegin = indices.begin() + range.first;
                        auto rangeEnd = indices.begin() + range.second;

                        auto middleIt = std::partition(rangeBegin, rangeEnd, [&](uint32_t id){
                            auto proj = hashes[id][depth];
                            return proj <= mid;
                        });

                        uint32_t rangeHalfSize = middleIt - rangeBegin;
                        Range lo = {range.first, range.first + rangeHalfSize};
                        Range hi = {range.first + rangeHalfSize , range.second};
                        {
                            std::lock_guard<std::mutex> guard(stack_mtx);
                            stack.emplace_back( depth+1, lo);
                            stack.emplace_back( depth+1, hi);
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

void constructResultSplitting(const vector<Vec>& points, vector<vector<uint32_t>>& result) {

    long timeBoundsMs;
    if(getenv("LOCAL_RUN")) {
        timeBoundsMs = 60'000;
    } else {
        timeBoundsMs = points.size() == 10'000 ? 20'000 : 1'600'000;
    }

    std::cout << "start run with time bound: " << timeBoundsMs << std::endl;

    auto startTime = hclock::now();
    vector<KnnSetScannable> idToKnn(points.size());
    uint32_t numPoints = points.size();

    uint32_t iteration = 0;
    while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
        std::cout << "Iteration: " << iteration << std::endl;
        auto startGroup = hclock::now();

        uint32_t numHashFuncs = requiredHashFuncs(points.size(), 300);
        std::vector<uint32_t> indices(numPoints);
        std::iota(indices.begin(), indices.end(), 0);
        vector<Range> ranges;
        splitHorizontalThread(numHashFuncs, points, ranges, indices);

        auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
        groupingTime += groupDuration;


        auto startProcessing = hclock::now();

        auto numThreads = std::thread::hardware_concurrency();
        vector<std::thread> threads;
        Task<Range> tasks(ranges);
        std::atomic<uint32_t> count = 0;
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                auto optRange = tasks.getTask();
                while (optRange) {
                    auto& range = *optRange;
                    uint32_t rangeSize = range.second - range.first;
                    count += rangeSize;
                    addCandidates(points, indices, range, idToKnn);
                    optRange= tasks.getTask();
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

    for (uint32_t id = 0; id < points.size(); ++id) {
        result[id] = idToKnn[id].finalize();
    }

    auto sizes = padResult(points, result);
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

