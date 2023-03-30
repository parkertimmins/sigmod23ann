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

struct SolutionKmeans {

    static inline std::atomic<uint64_t> groupingTime = 0;
    static inline std::atomic<uint64_t> processGroupsTime = 0;

    static void splitKmeans(uint32_t knnIterations, uint32_t maxGroupSize, uint32_t numPoints, float points[][104],
                            vector<Range> &ranges, vector<uint32_t> &indices) {

        auto startKnn = hclock::now();
        uint32_t branchingFactor = 2;
        auto numThreads = std::thread::hardware_concurrency();

        vector<std::thread> threads;

        // ranges are within the indices array, which contains ids
        vector<Range> stack;
        stack.emplace_back(make_pair(0, numPoints));
        std::mutex stack_mtx;
        std::mutex groups_mtx;

        std::atomic<uint32_t> count = 0;
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                while (count < numPoints) {
                    stack_mtx.lock();
                    if (stack.empty()) {
                        stack_mtx.unlock();
                    } else {
                        auto range = stack.back();
                        stack.pop_back();
                        stack_mtx.unlock();
                        uint32_t rangeSize = range.second - range.first;

                        if (rangeSize < maxGroupSize) {
                            count += rangeSize;
                            std::lock_guard<std::mutex> guard(groups_mtx);
                            ranges.push_back(range);
                        } else {
                            // knn splits

                            // pick 2 point ids
                            std::unordered_set<uint32_t> centerIds;
                            std::uniform_int_distribution<uint32_t> distribution(range.first, range.second - 1);
                            while (centerIds.size() < branchingFactor) {
                                centerIds.insert(distribution(rd));
                            }

                            // copy points into Vec objects
                            vector<Vec> centers(centerIds.size());
                            uint32_t c = 0;
                            for (auto id: centerIds) {
                                Vec &v = centers[c];
                                v.resize(dims);
                                std::memcpy(v.data(), points[id], dims);
                                c++;
                            }

                            for (auto iteration = 0; iteration < knnIterations; ++iteration) {
                                vector<vector<double>> sumOfGroups(branchingFactor);
                                vector<uint32_t> groupSizes(branchingFactor, 0);
                                for (auto &sums: sumOfGroups) {
                                    sums.resize(dims);
                                }

                                // measure distance from all points in group to each of the 2 points
                                for (uint32_t i = range.first; i < range.second; ++i) {
                                    uint32_t minDistCenterIdx = 0;
                                    float minDist = std::numeric_limits<float>::max();

                                    auto id = indices[i];
                                    auto &pt = points[id];
                                    for (uint32_t c = 0; c < branchingFactor; ++c) {
                                        Vec &center = centers[c];
                                        float dist = distance(pt, center.data());
                                        if (dist < minDist) {
                                            minDist = dist;
                                            minDistCenterIdx = c;
                                        }
                                    }

                                    groupSizes[minDistCenterIdx]++;
                                    auto &vecSums = sumOfGroups[minDistCenterIdx];
                                    for (uint32_t i = 0; i < dims; ++i) {
                                        vecSums[i] += pt[i];
                                    }
                                }

                                // recompute centers based on averages
                                for (uint32_t c = 0; c < branchingFactor; ++c) {
                                    for (uint32_t i = 0; i < dims; ++i) {
                                        centers[c][i] = sumOfGroups[c][i] / groupSizes[c];
                                    }
                                }
                            }

                            // compute final groups
                            vector<vector<uint32_t>> groups(branchingFactor);
                            for (uint32_t i = range.first; i < range.second; ++i) {
                                uint32_t minDistCenterIdx = 0;
                                float minDist = std::numeric_limits<float>::max();

                                auto id = indices[i];
                                auto &pt = points[id];
                                for (uint32_t c = 0; c < branchingFactor; ++c) {
                                    Vec &center = centers[c];
                                    float dist = distance(pt, center.data());
                                    if (dist < minDist) {
                                        minDist = dist;
                                        minDistCenterIdx = c;
                                    }
                                }
                                groups[minDistCenterIdx].push_back(id);
                            }

                            // build ranges
                            vector<Range> subRanges;
                            uint32_t start = range.first;
                            for (auto &group: groups) {
                                if (group.empty()) {
                                    continue;
                                }
                                uint32_t end = start + group.size();
                                subRanges.emplace_back(start, end);
                                for (uint32_t i = 0; i < group.size(); ++i) {
                                    indices[start + i] = group[i];
                                }
                                start = end;
                            }
                            {
                                std::lock_guard<std::mutex> guard(stack_mtx);
                                stack.insert(stack.end(), subRanges.begin(), subRanges.end());
                            }
                        }
                    }
                }

            });
        }
        for (auto &thread: threads) { thread.join(); }

#ifdef PRINT_OUTPUT
        std::cout << "group knn time: " << duration_cast<milliseconds>(hclock::now() - startKnn).count() << '\n';
#endif
    }

    static pair<Vec, Vec> kmeansStartVecs(Range& range, float points[][104], vector<uint32_t>& indices) {
        uint32_t rangeSize = range.second - range.first;
        uint32_t sampleSize = pow(log10(rangeSize), 3.5); // 129 samples for 10m bucket, 16 samples for bucket of 1220
        vector<uint32_t> idSample;
        idSample.reserve(sampleSize);
        std::uniform_int_distribution<uint32_t> distribution(range.first, range.second-1);
        while (idSample.size() < sampleSize) {
            idSample.push_back(indices[distribution(rd)]);
        }

        float maxDist = std::numeric_limits<float>::min();
        float* pii = nullptr;
        float* pjj = nullptr;
        for (uint32_t i = 0; i < idSample.size() - 1; ++i) {
            for (uint32_t j = i+1; j < idSample.size(); ++j) {
                float* pi = points[idSample[i]];
                float* pj = points[idSample[j]];
                float dist = distance(pi, pj);
                if (dist > maxDist) {
                    maxDist = dist;
                    pii = pi;
                    pjj = pj;
                }
            }
        }

        // copy points into Vec objects
        Vec center1(dims);
        Vec center2(dims);
        // TODO use copy as memcpy not safe
        for (uint32_t i = 0; i < dims; ++i) {
            center1[i]  = pii[i];
            center2[i]  = pjj[i];
        }
        return make_pair(center1, center2);
    }


    // handle both point vector data and array data
    static void splitKmeansBinaryTbb(Range range,
                              uint32_t knnIterations,
                              uint32_t maxGroupSize,
                              float points[][104],
                              vector<uint32_t>& indices,
                              tbb::concurrent_vector<Range>& completed
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            completed.push_back(range);
        } else {
            begin_kmeans:

            auto [center1, center2] = kmeansStartVecs(range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());
                // dot(x, coefs) >= offset means nearer to center1

                using centroid_agg = pair<uint32_t, vector<double>>;
                tbb::combinable<pair<centroid_agg, centroid_agg>> agg(make_pair(make_pair(0, vector<double>(100, 0.0f)), make_pair(0, vector<double>(100, 0.0f))));
                tbb::parallel_for(
                        tbb::blocked_range<uint32_t>(range.first, range.second),
                        [&](oneapi::tbb::blocked_range<uint32_t> r) {
                            auto& [agg1, agg2] = agg.local();
                            for (uint32_t i = r.begin(); i < r.end(); ++i) {
                                auto id = indices[i];
                                auto pt = std::begin(points[id]);
                                bool nearerCenter1 = dot(coefs.data(), pt) >= offset;
                                auto& aggToUse = nearerCenter1 ? agg1 : agg2;
                                aggToUse.first++;
                                for (uint32_t j = 0; j < dims; ++j) { aggToUse.second[j] += pt[j]; }
                            }
                        }
                );
                auto [c1_agg, c2_agg] = agg.combine([](const pair<centroid_agg, centroid_agg>& x, const pair<centroid_agg, centroid_agg>& y) {
                    centroid_agg c1{0, vector<double>(100, 0.0f)};
                    centroid_agg c2{0, vector<double>(100, 0.0f)};
                    c1.first = x.first.first + y.first.first;
                    c2.first = x.second.first + y.second.first;
                    for (uint32_t j = 0; j < dims; ++j) {
                        c1.second[j] = x.first.second[j] + y.first.second[j];
                        c2.second[j] = x.second.second[j] + y.second.second[j];
                    }
                    return make_pair(c1, c2);
                });

                if (c1_agg.first == 0 || c2_agg.first == 0) {
                    goto begin_kmeans;
                }

                // recompute centers based on averages
                for (uint32_t i = 0; i < dims; ++i) {
                    center1[i] = c1_agg.second[i] / c1_agg.first;
                    center2[i] = c2_agg.second[i] / c2_agg.first;
                }
            }

            // compute final groups
            auto between = scalarMult(0.5, add(center1, center2));
            auto coefs = sub(center1, between);
            auto offset = dot(between.data(), coefs.data());

            using groups = pair<vector<uint32_t>, vector<uint32_t>>;
            tbb::combinable<groups> groupsAgg(make_pair<>(vector<uint32_t>(), vector<uint32_t>()));
            tbb::parallel_for(
                    tbb::blocked_range<uint32_t>(range.first, range.second),
                    [&](tbb::blocked_range<uint32_t> r) {
                        auto& [g1, g2] = groupsAgg.local();
                        for (uint32_t i = r.begin(); i < r.end(); ++i) {
                            auto id = indices[i];
                            auto& pt = points[id];
                            bool nearerCenter1 = dot(coefs.data(), pt) >= offset;
                            auto& group = nearerCenter1 ? g1 : g2;
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
                goto begin_kmeans;
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
                [&]{ splitKmeansBinaryTbb(subRange1, knnIterations, maxGroupSize, points, indices, completed); },
                [&]{ splitKmeansBinaryTbb(subRange2, knnIterations, maxGroupSize, points, indices, completed); }
            );
        }
    }



    // handle both point vector data and array data
    static void splitKmeansSingleThreaded(Range range,
                              uint32_t knnIterations,
                              uint32_t maxGroupSize,
                              float points[][104],
                              vector<uint32_t>& indices,
                              vector<Range>& completed
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            completed.push_back(range);
        } else {
            begin_kmeans:

            auto [center1, center2] = kmeansStartVecs(range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());
                // dot(x, coefs) >= offset means nearer to center1

                using centroid_agg = pair<uint32_t, vector<double>>;
                centroid_agg c1 = make_pair(0, vector<double>(100, 0.0));
                centroid_agg c2 = make_pair(0, vector<double>(100, 0.0));
                for (uint32_t i = range.first; i < range.second; ++i) {
                    auto id = indices[i];
                    auto pt = std::begin(points[id]);
                    bool nearerCenter1 = dot(coefs.data(), pt) >= offset;
                    if (nearerCenter1) {
                        c1.first++;
                        for (uint32_t j = 0; j < dims; ++j) { c1.second[j] += pt[j]; }
                    } else {
                        c2.first++;
                        for (uint32_t j = 0; j < dims; ++j) { c2.second[j] += pt[j]; }
                    }

                }

                if (c1.first == 0 || c2.first == 0) {
                    goto begin_kmeans;
                }

                // recompute centers based on averages
                for (uint32_t i = 0; i < dims; ++i) {
                    center1[i] = c1.second[i] / c1.first;
                    center2[i] = c2.second[i] / c2.first;
                }
            }

            // compute final groups
            auto between = scalarMult(0.5, add(center1, center2));
            auto coefs = sub(center1, between);
            auto offset = dot(between.data(), coefs.data());

            vector<uint32_t> group1, group2;
            for (uint32_t i = range.first; i < range.second; ++i) {
                auto id = indices[i];
                auto& pt = points[id];
                bool nearerCenter1 = dot(coefs.data(), pt) >= offset;
                auto& group = nearerCenter1 ? group1 : group2;
                group.push_back(id);
            }

            if (group1.empty() || group2.empty()) {
                goto begin_kmeans;
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

            splitKmeansSingleThreaded(subRange1, knnIterations, maxGroupSize, points, indices, completed);
            splitKmeansSingleThreaded(subRange2, knnIterations, maxGroupSize, points, indices, completed);
        }
    }

    static void splitSortKnnForAdjacency(float pointsRead[][104], std::vector<uint32_t>& newToOldIndices, float points[][104], uint32_t numThreads, uint32_t numPoints, tbb::concurrent_vector<Range>& ranges) {
        auto startAdjacencySort = hclock::now();
        std::iota(newToOldIndices.begin(), newToOldIndices.end(), 0);
        splitKmeansBinaryTbb({0, numPoints}, 1, 400, pointsRead, newToOldIndices, ranges);
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
//        while (iteration < 150) {
        while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
    #ifdef PRINT_OUTPUT
            std::cout << "Iteration: " << iteration << '\n';
    #endif

            if (!first) {
                auto startGroup = hclock::now();
                splitKmeansBinaryTbb({0, numPoints}, 1, 400, points, indices, ranges);

                auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
                std::cout << "grouping time: " << groupDuration << '\n';
                groupingTime += groupDuration;
            }

            auto startProcessing = hclock::now();

            vector<std::thread> threads;
            std::atomic<uint32_t> count = 0;
            Task<Range, tbb::concurrent_vector<Range>> tasks(ranges);
            for (uint32_t t = 0; t < numThreads; ++t) {
                threads.emplace_back([&]() {
                    auto optRange = tasks.getTask();
                    while (optRange) {
                        auto& range = *optRange;
                        uint32_t rangeSize = range.second - range.first;
                        count += rangeSize;
                        addCandidates(points, indices, range, idToKnn);
                        optRange = tasks.getTask();
                    }
                });
            }

            for (auto& thread: threads) { thread.join(); }

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

    static void constructResultHighLevelParallelism(float points[][104], uint32_t numPoints, vector<vector<uint32_t>>& result) {

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

        vector<vector<uint32_t>> indicesPerThread(numThreads);
        for (auto& indices : indicesPerThread) {
            std::iota(indices.begin(), indices.end(), 0);
        }

        std::atomic<uint32_t> iteration = 0;
        vector<std::thread> threads;
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                vector<uint32_t> indices(numPoints);
                std::iota(indices.begin(), indices.end(), 0);
                vector<Range> ranges;

                while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
                    auto currIteration = iteration++;

                    auto startGroup = hclock::now();
                    splitKmeansSingleThreaded({0, numPoints}, 1, 400, points, indices, ranges);
                    auto groupTime = duration_cast<milliseconds>(hclock::now() - startGroup).count();

                    auto startProcess = hclock::now();
                    for (auto& range : ranges) {
                        addCandidatesThreadSafe(points, indices, range, idToKnn);
                    }
                    ranges.clear();
                    auto processTime = duration_cast<milliseconds>(hclock::now() - startProcess).count();

                    groupingTime += groupTime;
                    processGroupsTime += processTime;
                    std::cout << "iteration: " << currIteration << ", thread: " << t << ", group: " << groupTime << ", process: " << processTime << "\n";

                }
            });
        }

        for (auto& thread: threads) { thread.join(); }


        for (uint32_t id = 0; id < numPoints; ++id) {
            result[id] = idToKnn[id].finalize();
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
