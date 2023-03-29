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
#include <deque>
#include <emmintrin.h>
#include <immintrin.h>
#include <oneapi/tbb/concurrent_vector.h>
#include "oneapi/tbb.h"
#include "Constants.hpp"
#include "KnnSets.hpp"
#include "LinearAlgebra.hpp"
#include "Utility.hpp"


using std::cout;
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




uint64_t groupingTime = 0;
uint64_t processGroupsTime = 0;







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

uint32_t requiredHashFuncs4way(uint32_t numPoints, uint32_t maxBucketSize) {
    uint32_t groupSize = numPoints;
    uint32_t numHashFuncs = 0;
    while (groupSize > maxBucketSize) {
        groupSize /= 4;
        numHashFuncs++;
    }
    return numHashFuncs;
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



void splitHorizontalUniformSample(uint32_t numHashFuncs, uint32_t numPoints, float points[][104], std::unordered_map<uint64_t, vector<uint32_t>>& globalGroups) {
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


#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << '\n';
#endif
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

    vector<float> splits(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        auto& [min, max] = globalBounds[h];
        std::uniform_real_distribution<float> distribution(min, max);
        splits[h] = distribution(rd);
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
                    if (proj > splits[h]) {
                        sig |= (1 << h);
                    }
                }
                auto it = groups.find(sig);
                if (it == groups.end()) {
                    groups[sig] = {i};
                } else {
                    it->second.push_back(i);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }


    for (unordered_map<uint64_t, vector<uint32_t>>& localGroupSet: localGroups) {
        for (auto& [sig, group] : localGroupSet) {
            auto it = globalGroups.find(sig);
            if (it == globalGroups.end()) {
                 globalGroups[sig] = group;
            } else {
                auto& globalGroup = it->second;
                globalGroup.insert(globalGroup.end(), group.begin(), group.end());
            }
        }
    }
#ifdef PRINT_OUTPUT
    std::cout << "histogram split time: " << duration_cast<milliseconds>(hclock::now() - startSplit).count() << '\n';
#endif
}



void splitHorizontalMean(uint32_t numHashFuncs, uint32_t numPoints, float points[][104], std::unordered_map<uint64_t, vector<uint32_t>>& globalGroups) {
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


    auto startSplit = hclock::now();
#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(startSplit - startHash).count() << '\n';
#endif

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

    vector<float> meanSplits(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        auto& [min, max] = globalBounds[h];
        meanSplits[h] = (min + max) / 2;
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
                    float median = meanSplits[h];
                    if (proj > median) {
                        sig |= (1 << h);
                    }
                }
                auto it = groups.find(sig);
                if (it == groups.end()) {
                    groups[sig] = {i};
                } else {
                    it->second.push_back(i);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    for (unordered_map<uint64_t, vector<uint32_t>>& localGroupSet: localGroups) {
        for (auto& [sig, group] : localGroupSet) {

            auto it = globalGroups.find(sig);
            if (it == globalGroups.end()) {
                globalGroups[sig] = group;
            } else {
                auto& globalGroup = it->second;
                globalGroup.insert(globalGroup.end(), group.begin(), group.end());
            }
        }
    }

#ifdef PRINT_OUTPUT
    std::cout << "histogram split time: " << duration_cast<milliseconds>(hclock::now() - startSplit).count() << '\n';
#endif
}

pair<float, float> getBucketBounds(pair<float, float>& bounds, uint32_t numHistogramBuckets, uint32_t bucketId) {
    auto& [min, max] = bounds;
    float width = max - min;
    float bucketWidth = width / numHistogramBuckets;
    float bucketStart = bucketId * bucketWidth + min;
    float bucketEnd = bucketStart + bucketWidth;
    return {bucketStart, bucketEnd};
}



void splitHorizontalHistogram4way(uint32_t numHashFuncs, uint32_t numPoints, float points[][104], std::unordered_map<uint64_t, vector<uint32_t>>& globalGroups) {
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


#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << '\n';
#endif
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

    // compute quantile split points
    vector<vector<float>> splitsBounds(numHashFuncs);
    for (uint32_t h = 0; h < numHashFuncs; ++h) {
        uint32_t pointsSeen = 0;
        bool q1 = false, q2=false, q3=false;
        for (uint32_t b = 0; b < numHistogramBuckets; ++b) {
            pointsSeen += globalHistograms[h][b];
            // bucket contains median
            if (!q1 && pointsSeen >= numPoints / 4) {
                q1 = true;
                auto [bucketStart, bucketEnd] = getBucketBounds(globalBounds[h], numHistogramBuckets, b);
                std::uniform_real_distribution<float> distribution(bucketStart, bucketEnd);
                splitsBounds[h].push_back(distribution(rd));
            } else if (!q2 && pointsSeen >= numPoints / 2) {
                q2 = true;
                auto [bucketStart, bucketEnd] = getBucketBounds(globalBounds[h], numHistogramBuckets, b);
                std::uniform_real_distribution<float> distribution(bucketStart, bucketEnd);
                splitsBounds[h].push_back(distribution(rd));
            } else if (!q3 && pointsSeen >= 3 * numPoints / 4) {
                q3 = true;
                auto [bucketStart, bucketEnd] = getBucketBounds(globalBounds[h], numHistogramBuckets, b);
                std::uniform_real_distribution<float> distribution(bucketStart, bucketEnd);
                splitsBounds[h].push_back(distribution(rd));
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
                    vector<float> quantiles = splitsBounds[h];
                    if (proj < quantiles[0]) {
//                        sig |= 0;
                    } else if (proj < quantiles[1]) {
                        sig |= 1;
                    } else if (proj < quantiles[2]) {
                        sig |= 2;
                    } else {
                        sig |= 3;
                    }
                    sig <<= 2;
                }

                auto it = groups.find(sig);
                if (it == groups.end()) {
                    groups[sig] = {i};
                } else {
                    it->second.push_back(i);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    for (unordered_map<uint64_t, vector<uint32_t>>& localGroupSet: localGroups) {
        for (auto& [sig, group] : localGroupSet) {

            auto it = globalGroups.find(sig);
            if (it == globalGroups.end()) {
                globalGroups[sig] = group;
            } else {
                auto& globalGroup = it->second;
                globalGroup.insert(globalGroup.end(), group.begin(), group.end());
            }
        }
    }
#ifdef PRINT_OUTPUT
    std::cout << "histogram split time: " << duration_cast<milliseconds>(hclock::now() - startSplit).count() << '\n';
#endif
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


#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << '\n';
#endif
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

                auto it = groups.find(sig);
                if (it == groups.end()) {
                    groups[sig] = {i};
                } else {
                    it->second.push_back(i);
                }
            }
        });
    }
    for (auto& thread: threads) { thread.join(); }

    for (unordered_map<uint64_t, vector<uint32_t>>& localGroupSet: localGroups) {
        for (auto& [sig, group] : localGroupSet) {

            auto it = globalGroups.find(sig);
            if (it == globalGroups.end()) {
                globalGroups[sig] = group;
            } else {
                auto& globalGroup = it->second;
                globalGroup.insert(globalGroup.end(), group.begin(), group.end());
            }
        }
    }
#ifdef PRINT_OUTPUT
    std::cout << "histogram split time: " << duration_cast<milliseconds>(hclock::now() - startSplit).count() << '\n';
#endif
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

#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << '\n';
#endif

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

#ifdef PRINT_OUTPUT
    std::cout << "group regroup time: " << duration_cast<milliseconds>(hclock::now() - startRegroup).count() << '\n';
    std::cout << "group regroup maximum group size: " << actualMaxGroupsSize << '\n';
#endif
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

#ifdef PRINT_OUTPUT
    std::cout << "group hash time: " << duration_cast<milliseconds>(hclock::now() - startHash).count() << '\n';
#endif
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

#ifdef PRINT_OUTPUT
    std::cout << "group regroup time: " << duration_cast<milliseconds>(hclock::now() - startRegroup).count() << '\n';
#endif
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
#ifdef PRINT_OUTPUT
    std::cout << "adjacency sort time: " << adjacencySortDuration << '\n';
#endif
}





