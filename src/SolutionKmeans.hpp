#ifndef SIGMOD23ANN_SOLUTIONKMEANS_HPP
#define SIGMOD23ANN_SOLUTIONKMEANS_HPP

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
#include <ranges>
#include "../thirdparty/perfevent/PerfEvent.hpp"

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


#define PERF


struct SolutionKmeans {

    static inline std::atomic<uint64_t> groupProcessTime = 0;

    static float calcSamplePercent(uint32_t min, uint32_t max) {
        uint32_t rangeSize = max - min;
        return pow(log(rangeSize) / log(30), 7) / rangeSize; // 0.005 for 1e7, around 10% for 10k
    }


    static uint64_t topUpSingle(float points[][112], vector<KnnSetScannableSimd>& idToKnn) {
        auto startTopup = hclock::now();
        uint32_t numPoints = idToKnn.size();

        std::atomic<uint64_t> nodesUpdated = 0;
        std::atomic<uint64_t> nodesAdded = 0;

        vector<vector<uint32_t>> knnIds(numPoints);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numPoints),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                for (auto id = r.begin(); id < r.end(); ++id) {
                    vector<uint32_t> ids;
                    ids.reserve(100);
                    for (auto& id2 : idToKnn[id].current_ids) {
                        ids.push_back(id2);
                    }
                    std::sort(ids.begin(), ids.end());
                    knnIds[id] = std::move(ids);
                }
            }
        );

        std::cout << "top copy idKnnSet time: " << duration_cast<milliseconds>(hclock::now() - startTopup).count() << "\n";

        std::atomic<uint32_t> count = 0;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numPoints),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                tsl::robin_set<uint32_t> candidates;
                for (auto id1 = r.begin(); id1 < r.end(); ++id1) {
                    uint32_t added = 0;
                    auto& knn = knnIds[id1];
                    auto& knnSet = idToKnn[id1];
                    for (auto& id2 : knnIds[id1]) {
                        for (auto& id3 : knnIds[id2]) {
                            candidates.insert(id3);
                        }
                    }

                    // remove current ids and self id
//                    for (auto& id2 : knnIds[id1]) { candidates.erase(id2); }
                    candidates.erase(id1);

                    for (auto& id3 : candidates) {
                        float dist = distance(points[id3], points[id1]);
                        if (knnSet.addCandidate(id3, dist)) {
                            added++;
                        }
                    }
                    candidates.clear();

                    auto currCount = count++;
                    if (currCount % 10'000 == 0) {
                        auto topupTime = duration_cast<milliseconds>(hclock::now() - startTopup).count();
                        std::cout << "topped up: " << currCount << ", timing topping up:" << topupTime << "\n";
                    }
                    if (added > 0) {
                        nodesUpdated++;
                        nodesAdded += added;
                    }
                }
            }
        );

        std::cout << "topUp nodes added: " << nodesAdded << "\n";
        std::cout << "topUp nodes changed: " << nodesUpdated << "\n";
        return nodesUpdated.load();
    }

    static uint64_t topUp(float points[][112], vector<KnnSetScannable>& idToKnn) {
        auto startTopup = hclock::now();
        uint32_t numPoints = idToKnn.size();

        std::atomic<uint64_t> nodesUpdated = 0;
        std::atomic<uint64_t> nodesAdded = 0;

        vector<vector<pair<uint32_t, bool>>> knnIds(numPoints);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numPoints),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                for (auto id = r.begin(); id < r.end(); ++id) {
                    vector<pair<uint32_t, bool>> ids;
                    ids.reserve(100);
                    for (auto &[dist, id2, isNew] : idToKnn[id].queue) {
                        ids.emplace_back(id2, isNew);
                        isNew = false;
                    }
                    std::sort(ids.begin(), ids.end());
                    knnIds[id] = std::move(ids);
                }
            }
        );

        std::cout << "top copy idKnnSet time: " << duration_cast<milliseconds>(hclock::now() - startTopup).count() << "\n";

        std::atomic<uint32_t> count = 0;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numPoints),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                tsl::robin_set<uint32_t> candidates;
                for (auto id1 = r.begin(); id1 < r.end(); ++id1) {
                    uint32_t added = 0;
                    auto& knn = knnIds[id1];
                    auto& knnSet = idToKnn[id1];
                    for (auto& [id2, isNew2] : knnIds[id1]) {
                        if (isNew2) {
                            // if id2 is new, need to add all neighbors
                            for (auto& [id3, isNew3]: knnIds[id2]) { candidates.insert(id3); }
                        } else {
                            // if id2 is not new, only need to add its new neighbors
                            for (auto& [id3, isNew3]: knnIds[id2]) {
                                if (isNew3) { candidates.insert(id3); }
                            }
                        }
                    }

                    // remove current ids and self id
//                    for (auto& id2 : knnIds[id1]) { candidates.erase(id2); }
                    candidates.erase(id1);

                    for (auto& id3 : candidates) {
                        float dist = distance(points[id3], points[id1]);
                        if (knnSet.addCandidate(id3, dist)) {
                            added++;
                        }
                    }
                    candidates.clear();

                    auto currCount = count++;
                    if (currCount % 10'000 == 0) {
                        auto topupTime = duration_cast<milliseconds>(hclock::now() - startTopup).count();
                        std::cout << "topped up: " << currCount << ", timing topping up:" << topupTime << "\n";
                    }
                    if (added > 0) {
                        nodesUpdated++;
                        nodesAdded += added;
                    }
                }
            }
        );

        std::cout << "topUp nodes added: " << nodesAdded << "\n";
        std::cout << "topUp nodes changed: " << nodesUpdated << "\n";
        return nodesUpdated.load();
    }

    static vector<vector<uint32_t>> aggregateGroups(uint32_t numPoints, uint32_t numPossibleGroups, vector<uint32_t>& id_to_group) {
        uint32_t numThreads = std::thread::hardware_concurrency();
        auto ranges = splitRange({0, numPoints}, numThreads);

        auto s = hclock::now();
        // convert id->grpId into grpId -> {id}
        vector<vector<vector<uint32_t>>> localGrpToIds(ranges.size());
        vector<std::thread> threads;
        for (uint32_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                auto& local = localGrpToIds[t];
                local.resize(numPossibleGroups);
                auto range = ranges[t];
                for (uint32_t i = range.first; i < range.second; ++i) {
                    uint32_t grp = id_to_group[i];
                    local[grp].push_back(i);
                }
            });
        }
        for (auto& thread: threads) { thread.join(); }

        vector<vector<uint32_t>> globalGrpToIds(numPossibleGroups);

        auto numGroupingThreads = std::min(numThreads, numPossibleGroups);
        auto groupRanges = splitRange({0, numPossibleGroups}, numGroupingThreads);
        // convert id->grpId into grpId -> {id}
        threads.clear();
        for (uint32_t t = 0; t < numGroupingThreads; ++t) {
            threads.emplace_back([&, t]() {
                auto& gr = groupRanges[t];
                for (uint32_t g = gr.first; g < gr.second; ++g) {
                    auto& ids = globalGrpToIds[g];
                    for (auto &local: localGrpToIds) {
                        auto& idsLocal = local[g];
                        ids.insert(ids.end(), idsLocal.begin(), idsLocal.end());
                    }
                }
            });
        }
        for (auto& thread: threads) { thread.join(); }

        return globalGrpToIds;
    }

    static vector<pair<uint32_t, uint32_t>> getStartVecs(uint32_t idealGroupSize, float points[][112], uint32_t numPossibleGroups, vector<vector<uint32_t>>& grpIdToGroup) {
        vector<pair<uint32_t, uint32_t>> samples(numPossibleGroups);
        uint32_t numSamples = 10;
        for (uint32_t g = 0; g < numPossibleGroups; ++g) {
            auto& ids = grpIdToGroup[g];
            if (ids.size() <= idealGroupSize) {
                // signifies that group is too small to split
                samples[g] = {UINT32_MAX, UINT32_MAX};
            } else {
                std::uniform_int_distribution<uint32_t> distribution(0, ids.size() - 1);

                vector<uint32_t> sample;
                sample.reserve(numSamples);
                while (sample.size() < numSamples) {
                    sample.push_back(ids[distribution(rd)]);
                }

                float maxDist = 0;
                uint32_t idi, idj;
                for (uint32_t i = 0; i < numSamples - 1; ++i) {
                    for (uint32_t j = i + 1; j < numSamples; ++j) {
                        float dist = distance(points[sample[i]], points[sample[j]]);
                        if (dist > maxDist) {
                            maxDist = dist;
                            idi = sample[i];
                            idj = sample[j];
                        }
                    }
                }
                samples[g] = {idi, idj};
            }
        }
        return samples;
    }

    static uint32_t requiredHashFuncs(uint32_t numPoints, uint32_t maxBucketSize) {
        uint32_t groupSize = numPoints;
        uint32_t numHashFuncs = 0;
        while (groupSize > maxBucketSize) {
            groupSize /= 2;
            numHashFuncs++;
        }
        return numHashFuncs;
    }

    inline static vector<long> stage;

    // handle both point vector data and array data
    static vector<vector<uint32_t>> splitKmeansNonRec(
            uint32_t numPoints,
            uint32_t knnIterations,
            uint32_t idealGroupSize,
            float points[][112],
            vector<KnnSetScannableSimd>& idToKnn) {
        stage.resize(20);
        uint32_t numThreads = std::thread::hardware_concurrency();
        auto ranges = splitRange({0, numPoints}, numThreads);

        vector<uint32_t> id_to_group(numPoints, 0);

        uint32_t maxDepth = requiredHashFuncs(numPoints, idealGroupSize);
        uint32_t numCurrGroups = 1;

        vector<vector<uint32_t>> singleGroup(1);
        singleGroup[0].resize(numPoints);
        std::iota(singleGroup[0].begin(), singleGroup[0].end(), 0);

        for (uint32_t depth = 0; depth < maxDepth; ++depth) {

            // Get samples for initial centers
#ifdef PERF
            PerfEvent perf;
            perf.startCounters();
#endif
            auto s1 = hclock::now();
            auto grpIdToGroup = depth == 0 ? std::move(singleGroup) : aggregateGroups(numPoints, numCurrGroups, id_to_group);
            auto groupStarts = getStartVecs(idealGroupSize, points, numCurrGroups, grpIdToGroup);
            stage[1] += duration_cast<milliseconds>(hclock::now() - s1).count();

#ifdef PERF
            perf.stopCounters();
            std::cout << "depth " << depth;
            std::cout << "sample start points\n";
            perf.printReport(std::cout, 100'000);
            std::cout << std::endl;
            perf.startCounters();
#endif

            // get centers and split plane from samples
            auto s2 = hclock::now();
            vector<pair<Vec, Vec>> groupCenters(numCurrGroups);
            vector<pair<float, Vec>> groupPlanes(numCurrGroups);;
            vector<bool> shouldSplit(numCurrGroups, true);
            for (uint32_t g = 0; g < numCurrGroups; ++g) {
                auto& starts = groupStarts[g];
                if (starts.first == UINT32_MAX) {
                    shouldSplit[g] = false;
                } else {
                    Vec c1(100);
                    Vec c2(100);
                    for (uint32_t j = 0; j < 100; ++j) { c1[j] = points[starts.first][j]; }
                    for (uint32_t j = 0; j < 100; ++j) { c2[j] = points[starts.second][j]; }
                    groupCenters[g] = { c1, c2 };

                    auto between = scalarMult(0.5, add(c1, c2));
                    auto coefs = sub(c1, between);
                    auto offset = dot(between.data(), coefs.data());
                    groupPlanes[g] = { offset, coefs };
                }
            };
            stage[2] += duration_cast<milliseconds>(hclock::now() - s2).count();

#ifdef PERF
            perf.stopCounters();
            std::cout << "depth " << depth;
            std::cout << "compute split planes\n";
            perf.printReport(std::cout, 100'000);
            std::cout << std::endl;
            perf.startCounters();
#endif

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {

                uint32_t expGroupsAtDepth = 1 << depth;
                uint32_t expGroupSize = numPoints / expGroupsAtDepth;
                float sampleRate = calcSamplePercent(0, expGroupSize);
//                std::cout << "depth: " << depth << ", exp groups: " << expGroupsAtDepth << ", exp group size: " << expGroupSize << ", sample rate: " << sampleRate << "\n";


                // assign points to either side of splits planes
                auto s3 = hclock::now();
                using centroid_agg = pair<uint32_t, std::array<float, 100>>;
                vector<vector<pair<centroid_agg, centroid_agg>>> localGroupCenterAggs(ranges.size());
                vector<std::thread> threads;
                for (uint32_t t = 0; t < numThreads; ++t) {
                    threads.emplace_back([&, t]() {
                        auto range = ranges[t];
                        std::uniform_real_distribution<float> sampleDist(0, 1);
                        auto& center_agg = localGroupCenterAggs[t];
                        center_agg.resize(numCurrGroups, { { 0, std::array<float, 100>() }, { 0, std::array<float, 100>() } });
                        for (uint32_t i = range.first; i < range.second; ++i) {
                            // !could result in some groups never getting sampled!
                            auto grp = id_to_group[i];
                            if (shouldSplit[grp]) {
                                if (sampleDist(rd) < sampleRate) {
                                    auto& [offset, coefs] = groupPlanes[grp];
                                    auto& pt = points[i];
                                    auto& [agg1, agg2] = center_agg[grp];
                                    auto& aggToUse = dot(coefs.data(), pt) >= offset ? agg1 : agg2;
                                    aggToUse.first++;
                                    plusEq(aggToUse.second.data(), pt);
                                }
                            }
                        }
                    });
                }
                for (auto& thread: threads) { thread.join(); }
                stage[3] += duration_cast<milliseconds>(hclock::now() - s3).count();


                // aggregate local results for split plane assignment
                auto s4 = hclock::now();
                vector<pair<centroid_agg, centroid_agg>> globalCenterAggs(numCurrGroups, { { 0, std::array<float, 100>() }, { 0, std::array<float, 100>() } });
                for (auto& local : localGroupCenterAggs) {
                    for (uint32_t g = 0; g < numCurrGroups; ++g) {
                        if (shouldSplit[g]) {
                            auto& ca = local[g];
                            auto& [c1, c2] = globalCenterAggs[g];
                            c1.first += ca.first.first;
                            c2.first += ca.second.first;
                            for (uint32_t j = 0; j < dims; ++j) { c1.second[j] += ca.first.second[j]; }
                            for (uint32_t j = 0; j < dims; ++j) { c2.second[j] += ca.second.second[j]; }
                        }
                    }
                }
                stage[4] += duration_cast<milliseconds>(hclock::now() - s4).count();

                auto s5 = hclock::now();
                for (uint32_t g = 0; g < numCurrGroups; ++g) {
                    if (shouldSplit[g]) {
                        auto& center_aggs = globalCenterAggs[g];
                        auto& [center1, center2] = groupCenters[g];
                        auto& [c1_agg, c2_agg] = center_aggs;

                        if (c1_agg.first == 0 || c2_agg.first == 0) {
                            continue;
                        }
                        for (uint32_t i = 0; i < dims; ++i) {
                            center1[i] = c1_agg.second[i] / c1_agg.first;
                            center2[i] = c2_agg.second[i] / c2_agg.first;
                        }
                        auto between = scalarMult(0.5, add(center1, center2));
                        auto coefs = sub(center1, between);
                        auto offset = dot(between.data(), coefs.data());
                        groupPlanes[g] = { offset, coefs };
                    }
                };
                stage[5] += duration_cast<milliseconds>(hclock::now() - s5).count();
            }


#ifdef PERF
            perf.stopCounters();
            std::cout << "depth " << depth;
            std::cout << "kmean iter loop\n";
            perf.printReport(std::cout, 100'000);
            std::cout << std::endl;
            perf.startCounters();
#endif

            // recompute groups based on plane
            auto s6 = hclock::now();
            vector<std::thread> threads;
            for (uint32_t t = 0; t < numThreads; ++t) {
                threads.emplace_back([&, t]() {
                    auto range = ranges[t];
                    for (uint32_t i = range.first; i < range.second; ++i) {
                        uint32_t grp = id_to_group[i];
                        if (shouldSplit[grp]) {
                            auto& [offset, coefs] = groupPlanes[grp];
                            id_to_group[i] = dot(coefs.data(), points[i]) >= offset ? grp : grp + numCurrGroups;
                        }
                    }
                });
            }
            for (auto& thread: threads) { thread.join(); }
            stage[6] += duration_cast<milliseconds>(hclock::now() - s6).count();

#ifdef PERF
            perf.stopCounters();
            std::cout << "depth " << depth;
            std::cout << "group split\n";
            perf.printReport(std::cout, 100'000);
            std::cout << std::endl;
            perf.startCounters();
#endif

            numCurrGroups *= 2;
        }

        // convert id->grpId into grpId -> {id}
        auto s7 = hclock::now();
        auto globalGrpToIds = aggregateGroups(numPoints, numCurrGroups, id_to_group);
        stage[7] += duration_cast<milliseconds>(hclock::now() - s7).count();

        vector<vector<uint32_t>> groups;
        groups.reserve(globalGrpToIds.size());

        uint32_t groupsSkipped = 0;
        uint32_t emptyGroups = 0;
        uint32_t totalIdsSkipped = 0;
        uint32_t idsToProcess = 0;
        for (uint32_t g = 0; g < numCurrGroups; ++g) {
            auto& group = globalGrpToIds[g];
            if (group.empty()) {
                emptyGroups++;
            } else if (group.size() < 2'000) {
                groups.push_back(group);
                idsToProcess += group.size();
            } else {
                groupsSkipped++;
                totalIdsSkipped+=group.size();
            }
        }
        std::cout << "num groups: " << groups.size() << "\n";
        std::cout << "num skipped groups: " << groupsSkipped << ", total size: " << totalIdsSkipped
            << ", avg size: " << static_cast<float>(totalIdsSkipped) / groupsSkipped << ", empty: " << emptyGroups <<  ", to process: " << idsToProcess << "\n";
        globalGrpToIds.clear();
        return groups;
    }

    static void constructResult(float points[][112], uint32_t numPoints, vector<vector<uint32_t>>& result) {

        bool localRun = getenv("LOCAL_RUN");
        auto numThreads = std::thread::hardware_concurrency();
        long timeBoundsMs = (localRun || numPoints == 10'000)  ? 20'000 : 1'650'000;



        std::cout << "start run with time bound: " << timeBoundsMs << '\n';

        auto startTime = hclock::now();
        vector<KnnSetScannableSimd> idToKnn(numPoints);

        uint32_t iteration = 0;
//        while (iteration < 10) {
        while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
            std::cout << "Iteration: " << iteration << '\n';

            auto startGroup = hclock::now();
            auto groups = splitKmeansNonRec(numPoints, 1, 400, points, idToKnn);
            auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
            std::cout << " group time: " << groupDuration << '\n';

            auto startProcess = hclock::now();
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(0, groups.size()),
                [&](tbb::blocked_range<uint32_t> r) {
                    for (uint32_t i = r.begin(); i < r.end(); ++i) {
                        addCandidatesGroup(points, groups[i], idToKnn);
                    }
                }
            );
            auto processDuration = duration_cast<milliseconds>(hclock::now() - startProcess).count();
            std::cout << " process time: " << processDuration << '\n';

            for (uint32_t i = 0; i < stage.size(); ++i) {
                if (stage[i] > 0)  {
                    std::cout << "s"  << i <<  ": " << stage[i] << "\n";
                    stage[i] = 0;
                }
            }

            iteration++;
        }

//        topUpSingle(points, idToKnn);

        for (uint32_t id = 0; id < numPoints; ++id) {
            result[id] = idToKnn[id].finalize();
        }

        auto sizes = padResult(numPoints, result);

        for (uint32_t i=0; i < sizes.size(); ++i) {
            std::cout << "size: " << i << ", count: " << sizes[i] << '\n';
        }
        std::cout << "total grouping/process time (ms): " << groupProcessTime << '\n';
    }




};
#endif //SIGMOD23ANN_SOLUTIONKMEANS_HPP
