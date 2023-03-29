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
using hclock = std::chrono::high_resolution_clock;
using std::unordered_map;
using std::make_pair;
using std::pair;
using std::vector;

struct SolutionRandomKD {

    static inline uint64_t groupingTime = 0;
    static inline uint64_t processGroupsTime = 0;

    // can have duplicates
    static vector<uint32_t> samplesIds(Range& range, vector<uint32_t>& indices) {
        uint32_t rangeSize = range.second - range.first;
        uint32_t sampleSize = pow(log10(rangeSize), 3.5); // 907 samples for 10m bucket, 36 samples for bucket with 610
        vector<uint32_t> sample;
        sample.reserve(sampleSize);
        std::uniform_int_distribution<uint32_t> distribution(range.first, range.second-1);
        while (sample.size() < sampleSize) {
            auto rangeIdx = distribution(rd);
            auto idx = indices[rangeIdx];
            sample.push_back(idx);
        }
        return sample;
    }

    static Vec estimatePca1(vector<uint32_t>& sample, float points[][104]) {
        float maxDist = std::numeric_limits<float>::min();
        float* pii = nullptr;
        float* pjj = nullptr;
        for (uint32_t i = 0; i < sample.size() - 1; ++i) {
            for (uint32_t j = i+1; j < sample.size(); ++j) {
                float* pi = points[sample[i]];
                float* pj = points[sample[j]];
                float dist = distance(pi, pj);
                if (dist > maxDist) {
                    maxDist = dist;
                    pii = pi;
                    pjj = pj;
                }
            }
        }

        return normalize(sub(pii, pjj));
    }

    // handle both point vector data and array data
    static void split(Range range,
                      uint32_t maxGroupSize,
                      float points[][104],
                      vector<uint32_t>& indices,
                      tbb::concurrent_vector<Range>& completed
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            completed.push_back(range);
        } else {
        begin_split:
            auto sample = samplesIds(range, indices);
            Vec pca1 = estimatePca1(sample, points);

            vector<pair<float, float*>> projectionValues;
            projectionValues.reserve(sample.size());
            for (auto& id : sample) {
                float* pt = points[id];
                auto proj = dot(pt, pca1.data());
                projectionValues.emplace_back(proj, pt);
            }
            std::sort(projectionValues.begin(), projectionValues.end());
            auto median = projectionValues[sample.size() / 2].first;

            // compute final groups
            using groups = pair<vector<uint32_t>, vector<uint32_t>>;
            tbb::combinable<groups> groupsAgg(make_pair<>(vector<uint32_t>(), vector<uint32_t>()));
            tbb::parallel_for(
                    tbb::blocked_range<uint32_t>(range.first, range.second, 1000),
                    [&](tbb::blocked_range<uint32_t> r) {
                        auto& [g1, g2] = groupsAgg.local();
                        for (uint32_t i = r.begin(); i < r.end(); ++i) {
                            auto id = indices[i];
                            auto& pt = points[id];
                            auto& group = dot(pca1.data(), pt) >= median ? g1 : g2;
                            group.push_back(id);
                        }
                    }
            );
            auto [group1, group2] = groupsAgg.combine([](const groups& x, const groups& y) {
                vector<uint32_t> g1;
                vector<uint32_t> g2;
                g1.insert(g1.end(), x.first.begin(), x.first.end());
                g1.insert(g1.end(), y.first.begin(), y.first.end());
                g2.insert(g2.end(), x.second.begin(), x.second.end());
                g2.insert(g2.end(), y.second.begin(), y.second.end());
                return make_pair(g1, g2);
            });

            if (group1.empty() || group2.empty()) {
                goto begin_split;
            }

            // build ranges
            uint32_t subRange1Start = range.first;
            uint32_t subRange2Start = range.first + group1.size();
            Range subRange1 = {subRange1Start, subRange1Start + group1.size()};
            Range subRange2 = {subRange2Start, subRange2Start + group2.size()};

            auto it1 = indices.data() + subRange1Start;
            std::memcpy(it1, group1.data(), group1.size() * sizeof(uint32_t));
            auto it2 = indices.data() + subRange2Start;
            std::memcpy(it2, group2.data(), group2.size() * sizeof(uint32_t));

            tbb::parallel_invoke(
                [&]{ split(subRange1, maxGroupSize, points, indices, completed); },
                [&]{ split(subRange2, maxGroupSize, points, indices, completed); }
            );
        }
    }

    static void splitSortKnnForAdjacency(float pointsRead[][104], std::vector<uint32_t>& newToOldIndices, float points[][104], uint32_t numThreads, uint32_t numPoints, tbb::concurrent_vector<Range>& ranges) {
        auto startAdjacencySort = hclock::now();
        std::iota(newToOldIndices.begin(), newToOldIndices.end(), 0);
        split({0, numPoints}, 400, pointsRead, newToOldIndices, ranges);
        vector<Range> moveRanges = splitRange({0, numPoints}, numThreads);
        vector<std::thread> threads;
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                auto moveRange = moveRanges[t];
                for (uint32_t newIdx = moveRange.first; newIdx < moveRange.second; ++newIdx) {
                    auto oldIdx = newToOldIndices[newIdx];
                    memcpy(points[newIdx], pointsRead[oldIdx], sizeof(float) * dims);
                }
            });
        }
        for (auto& thread: threads) { thread.join(); }

        delete[](pointsRead);
        auto adjacencySortDuration = duration_cast<milliseconds>(hclock::now() - startAdjacencySort).count();
    #ifdef PRINT_OUTPUT
        std::cout << "adjacency sort time: " << adjacencySortDuration << '\n';
    #endif
    }

    static void constructResult(float pointsRead[][104], uint32_t numPoints, vector<vector<uint32_t>>& result) {

        long timeBoundsMs;
        if(getenv("LOCAL_RUN")) {
            timeBoundsMs = 60'000;
        } else {
            timeBoundsMs = numPoints == 10'000 ? 20'000 : 1'650'000;
        }

    #ifdef PRINT_OUTPUT
        std::cout << "start run with time bound: " << timeBoundsMs << '\n';
    #endif
        auto startTime = hclock::now();
        vector<KnnSetScannable> idToKnn(numPoints);
        auto numThreads = std::thread::hardware_concurrency();

        // rewrite point data in adjacent memory and sort in a group order
        tbb::concurrent_vector<Range> ranges;
        std::vector<uint32_t> newToOldIndices(numPoints);
        float (*points)[104] = reinterpret_cast<float(*)[104]>(new __m256[(numPoints * 104 * sizeof(float)) / sizeof(__m256)]);
        splitSortKnnForAdjacency(pointsRead, newToOldIndices, points, numThreads, numPoints, ranges);
        std::vector<uint32_t> indices = newToOldIndices;

        bool first = true;

        uint32_t iteration = 0;
        while (iteration < 150) {
            //while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
    #ifdef PRINT_OUTPUT
            std::cout << "Iteration: " << iteration << '\n';
    #endif

            if (!first) {
                auto startGroup = hclock::now();
                split({0, numPoints}, 400, points, indices, ranges);

                auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
                std::cout << "grouping time: " << groupDuration << '\n';
                groupingTime += groupDuration;
            }

            auto startProcessing = hclock::now();

            vector<std::thread> threads;
            std::atomic<uint32_t> count = 0;
            Task<Range, tbb::concurrent_vector<Range>> tasks(ranges);
//            for (uint32_t t = 0; t < numThreads; ++t) {
//                threads.emplace_back([&]() {
//                    auto optRange = tasks.getTask();
//                    while (optRange) {
//                        auto& range = *optRange;
//                        uint32_t rangeSize = range.second - range.first;
//                        count += rangeSize;
//                        addCandidates(points, indices, range, idToKnn);
//                        optRange = tasks.getTask();
//                    }
//                });
//            }

//            for (auto& thread: threads) { thread.join(); }

            auto processingDuration = duration_cast<milliseconds>(hclock::now() - startProcessing).count();
            processGroupsTime += processingDuration;

    #ifdef PRINT_OUTPUT
            std::cout << "processing time: " << processingDuration << '\n';
            std::cout << "--------------------------------------------------------------------------------------------------------\n";
    #endif
            ranges.clear();
            first = false;
            iteration++;
        }

        for (uint32_t id = 0; id < numPoints; ++id) {
            auto newIdxResultRow = idToKnn[id].finalize();
            for (auto& ni : newIdxResultRow) {
                ni = newToOldIndices[ni];
            }
            result[newToOldIndices[id]] = std::move(newIdxResultRow);
        }

        auto sizes = padResult(numPoints, result);

    #ifdef PRINT_OUTPUT
        for (uint32_t i=0; i < sizes.size(); ++i) {
            std::cout << "size: " << i << ", count: " << sizes[i] << '\n';
        }
        std::cout << "total grouping time (ms): " << groupingTime << '\n';
        std::cout << "total processing time (ms): " << processGroupsTime << '\n';
    #endif
    }
};
