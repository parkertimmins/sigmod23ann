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
#include <unordered_set>
#include <cstdint>
#include <chrono>
#include <thread>
#include <optional>
#include <mutex>
#include <limits>
#include <queue>
#include "io.h"


#include <immintrin.h>
using namespace std;

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::seconds;
using Vec = vector<float>;
using hclock = std::chrono::high_resolution_clock;




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
 */

// DEPRECATED
uint32_t numProjections = 4;
uint8_t numBuckets = 4; //std::numeric_limits<unsigned char>::max();


uint64_t maxGroupSize = 1000; // will be around 600

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

float norm(Vec& vec) {
    float sumSquares = 0.0;
    for (auto& v : vec) {
        sumSquares += (v * v);
    }
    return sqrt(sumSquares);
}


std::default_random_engine rd(123);
vector<float> randUniformUnitVec(size_t dim=100) {
    std::mt19937 gen(rd());
    std::normal_distribution<> normalDist(0, 1);

    Vec randVec;
    randVec.reserve(dim);
    while (randVec.size() < dim) {
        auto elementValue = normalDist(gen);
        randVec.emplace_back(elementValue);
    }

    // make unit vector just for good measure
    float vecNorm = norm(randVec);
    for (float& v : randVec) {
        v = v / vecNorm;
    }
    return randVec;
}





Vec scalerMult(float c, const Vec& vec) {
    Vec result;
    for (auto& v : vec) {
        result.push_back(v * c);
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

// onto unit vector
Vec project(const Vec& randUnit, const Vec& vec) {
    return scalerMult(dot(randUnit, vec), randUnit);
}

uint64_t alpha = 10;
uint64_t makeSignature(const Vec& randUnit, const Vec& vec) {
    return dot(randUnit, vec) / alpha;
}

/*unordered_map<pair<uint32_t, uint32_t>, float> distCache;
float distanceCached(const vector<vector<float>> &points, uint32_t id1, uint32_t id2) {
    uint64_t pair = id1 < id2 ? ((static_cast<long>(id1) << 32) | id2) : ((static_cast<long>(id2) << 32) | id1);
    auto iter = distCache.find(pair);
    if (iter == distCache.end()) {
        float dist = distance128(points[id1], points[id1]);
        distCache[pair] = dist;
        return dist;
    } else {
        return iter->second;
    }
}*/

unordered_map<uint64_t, float> distCache;
float distanceCached(const vector<vector<float>> &points, uint32_t id1, uint32_t id2) {
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

vector<uint32_t> CalculateOneKnn(const vector<vector<float>> &data,
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

struct KnnSet {
    std::priority_queue<std::pair<float, uint32_t>> queue;
    std::unordered_set<uint32_t> set;
};

vector<uint32_t> finalize(KnnSet& currKnn) {
    vector<uint32_t> knn;
    while (!currKnn.queue.empty()) {
        knn.emplace_back(currKnn.queue.top().second);
        currKnn.queue.pop();
    }
    std::reverse(knn.begin(), knn.end());
    return knn;
}

void addCandidateToKnnSet(const uint32_t id, KnnSet& currKnn, const uint32_t candidate_id, float dist) {
    float lower_bound = std::numeric_limits<float>::max();
    if (id == candidate_id) return;  // skip itself.
    if (currKnn.set.find(candidate_id) != currKnn.set.end()) return; // already in set

    // only keep the top 100
    if (currKnn.queue.size() < 100 || dist < lower_bound) {
        currKnn.set.insert(candidate_id);
        currKnn.queue.push(std::make_pair(dist, candidate_id));
        if (currKnn.queue.size() > 100) {
            pair<float, uint32_t> toRemove = currKnn.queue.top();
            currKnn.set.erase(toRemove.second);
            currKnn.queue.pop();
        }
        lower_bound = currKnn.queue.top().first;
    }
}

void addCandidates(const vector<vector<float>> &points,
                   const vector<pair<float, uint32_t>>& group,
                   vector<KnnSet>& idToKnn) {
    for (uint32_t i=0, groupLen=group.size(); i < groupLen-1; ++i) {
        for (uint32_t j=i+1; j < groupLen; ++j) {
            auto id1 = group[i].second;
            auto id2 = group[j].second;
            float dist = distance128(points[id1], points[id2]);

            auto& knnSet1 = idToKnn[id1];
            auto& knnSet2 = idToKnn[id2];

            addCandidateToKnnSet(id1, knnSet1, id2, dist);
            addCandidateToKnnSet(id2, knnSet2, id1, dist);
        }
    }
}

void constructResultActual(const vector<Vec>& data, vector<vector<uint32_t>>& result) {

    result.resize(data.size());

    vector<uint32_t> sample_indexes(data.size());
    iota(sample_indexes.begin(), sample_indexes.end(), 0);

    auto numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

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



template<class K, class V>
vector<K> getKeys(std::unordered_map<K, V>& map) {
    vector<K> keys;
    for(const auto& kv : map) {
        keys.push_back(kv.first);
    }
    return keys;
}



vector<Vec> buildProjectionVecs(auto n) {
    vector<Vec> randVecs;
    while (randVecs.size() < n) {
        randVecs.push_back(randUniformUnitVec(100));
    }
    return randVecs;
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

    Task<uint64_t> tasks(getKeys(sigToPoints));

    auto numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

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


void print(vector<float>& ts) {
    for (float t: ts)
        std::cout << t << ", ";
    std::cout << std::endl;
}


void rehash(vector<pair<float, uint32_t>>& group, const vector<Vec>& points) {
    auto u = randUniformUnitVec(100);
    for (auto& p : group) {
        p.first = dot(u, points[p.second]);
    }
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

vector<pair<float, uint32_t>> buildInitialGroups(const vector<Vec>& points) {
    vector<pair<float, uint32_t>> group;
    auto numPoints = points.size();
    auto u = randUniformUnitVec(100);
    for (uint32_t i = 0; i < numPoints; ++i) {
        float hash = dot(u, points[i]);
        group.emplace_back(hash, i);
    }
    sort(group.begin(), group.end());
    return group;
}



void constructResultSplitting(const vector<Vec>& points, vector<vector<uint32_t>>& result) {
    auto startTime = hclock::now();
    vector<KnnSet> idToKnn(points.size());
    for (auto iteration = 0; iteration < 5; ++iteration) {
        std::cout << "start iteration: " << iteration << ", time (s): " << duration_cast<seconds>(hclock::now() - startTime).count() << std::endl;
        auto group1 = buildInitialGroups(points);
        vector<vector<pair<float, uint32_t>>> groups;

        auto startSort = hclock::now();
        splitRecursiveSingleThreaded(points, group1, groups);
        std::cout << "split/sort time: " << duration_cast<milliseconds>(hclock::now() - startSort).count() << std::endl;
        std::cout << "end sort iteration: " << iteration << ", time (s): " << duration_cast<seconds>(hclock::now() - startTime).count() << std::endl;

        auto numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        Task<vector<pair<float, uint32_t>>> tasks(groups);

        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                auto optGroup = tasks.getTask();
                while (optGroup) {
                    auto& grp = *optGroup;
//                    std::cout << "processing groups, size: " << grp.size() << std::endl;
                    addCandidates(points, grp, idToKnn);
                    optGroup = tasks.getTask();
                }
            });
        }

        for (auto& thread: threads) { thread.join(); }
        std::cout << "end group processing iteration: " << iteration << ", time (s): " << duration_cast<seconds>(hclock::now() - startTime).count() << std::endl;

    }

    for (uint32_t id = 0; id < points.size(); ++id) {
        result[id] = finalize(idToKnn[id]);
    }

    auto sizes = padResult(points, result);
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
  vector<vector<uint32_t>> knng(nodes.size());
  constructResultSplitting(nodes, knng);

  // Save to ouput.bin
  SaveKNNG(knng);

  return 0;
}

