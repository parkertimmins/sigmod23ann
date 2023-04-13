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

    static inline std::atomic<uint64_t> groupProcessTime = 0;

    static float calcSamplePercent(uint32_t min, uint32_t max) {
        uint32_t rangeSize = max - min;
        return pow(log(rangeSize) / log(30), 7) / rangeSize; // 0.005 for 1e7, around 10% for 10k
    }

    static vector<uint32_t> getSampleFromPercent(float perc, uint32_t min, uint32_t max) {
        uint32_t rangeSize = max - min;
        uint32_t sampleSize = rangeSize * perc;
        vector<uint32_t> sample;
        sample.reserve(sampleSize);
        std::uniform_int_distribution<uint32_t> distribution(min, max-1);
        while (sample.size() < sampleSize) {
            sample.push_back(distribution(rd));
        }
//        std::sort(sample.begin(), sample.end());
        return sample;
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
        for (auto& local : localGrpToIds) {
            for (uint32_t g = 0; g < numPossibleGroups; ++g) {
                auto& ids = globalGrpToIds[g];
                auto& idsLocal = local[g];
                ids.insert(ids.end(), idsLocal.begin(), idsLocal.end());
            }
        }
        return globalGrpToIds;
    }

    static vector<vector<uint32_t>> getPerGroupsSample(uint32_t numPoints, uint32_t numPossibleGroups, vector<uint32_t>& id_to_group) {
        auto s = hclock::now();
        auto globalGrpToIds = aggregateGroups(numPoints, numPossibleGroups, id_to_group);
        stage[12] += duration_cast<milliseconds>(hclock::now() - s).count();

        s = hclock::now();
        vector<vector<uint32_t>> samples(numPossibleGroups);
        for (uint32_t g = 0; g < numPossibleGroups; ++g) {
            auto& ids = globalGrpToIds[g];
            if (ids.empty()) { continue; }

            std::uniform_int_distribution<uint32_t> distribution(0, ids.size() - 1);
            // stupid case that should be fixed somewhere else!
            if (ids.size() <= 5) {
                samples[g] = ids.size() == 1 ? vector{ids[0], ids[0]} : vector{ids[0], ids[1]};
            } else {
                uint32_t id1 = ids[distribution(rd)];
                uint32_t id2;
                do {
                    id2 = ids[distribution(rd)];
                } while (id1 == id2);

                samples[g] = {id1, id2};
            }
        }
        stage[13] += duration_cast<milliseconds>(hclock::now() - s).count();

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
            uint32_t maxGroupSize,
            float points[][112],
            vector<KnnSetScannableSimd>& idToKnn) {
        stage.resize(20);
        uint32_t numThreads = std::thread::hardware_concurrency();
        auto ranges = splitRange({0, numPoints}, numThreads);

        vector<uint32_t> id_to_group(numPoints, 0);

        uint32_t maxDepth = requiredHashFuncs(numPoints, maxGroupSize);
        uint32_t numCurrGroups = 1;
        uint32_t numNextGroups = 2 * numCurrGroups;
        for (uint32_t depth = 0; depth < maxDepth; ++depth) {

            // Get samples for initial centers
            auto s1 = hclock::now();
            auto groupSamples = getPerGroupsSample(numPoints, numCurrGroups, id_to_group);
            stage[1] += duration_cast<milliseconds>(hclock::now() - s1).count();

            // get centers and split plane from samples
            auto s2 = hclock::now();
            vector<pair<Vec, Vec>> groupCenters(numCurrGroups);
            vector<pair<float, Vec>> groupPlanes(numCurrGroups);;
            for (uint32_t g = 0; g < numCurrGroups; ++g) {
                auto& sample = groupSamples[g];
                if (sample.empty()) { continue; }

                Vec c1(100);
                Vec c2(100);
                for (uint32_t j = 0; j < 100; ++j) { c1[j] = points[sample[0]][j]; }
                for (uint32_t j = 0; j < 100; ++j) { c2[j] = points[sample[1]][j]; }
                groupCenters[g] = { c1, c2 };

                auto between = scalarMult(0.5, add(c1, c2));
                auto coefs = sub(c1, between);
                auto offset = dot(between.data(), coefs.data());
                groupPlanes[g] = { offset, coefs };
            };
            stage[2] += duration_cast<milliseconds>(hclock::now() - s2).count();

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {

                uint32_t expGroupsAtDepth = 1 << depth;
                uint32_t expGroupSize = numPoints / expGroupsAtDepth;
                float sampleRate = calcSamplePercent(0, expGroupSize);
//                std::cout << "depth: " << depth << ", exp groups: " << expGroupsAtDepth << ", exp group size: " << expGroupSize << ", sample rate: " << sampleRate << "\n";


                // assign points to either side of splits planes
                auto s3 = hclock::now();
                using centroid_agg = pair<uint32_t, vector<float>>;
                vector<vector<pair<centroid_agg, centroid_agg>>> localGroupCenterAggs(ranges.size());
                vector<std::thread> threads;
                for (uint32_t t = 0; t < numThreads; ++t) {
                    threads.emplace_back([&, t]() {
                        auto range = ranges[t];
                        std::uniform_real_distribution<float> sampleDist(0, 1);
                        auto& center_agg = localGroupCenterAggs[t];
                        center_agg.resize(numCurrGroups, { { 0, vector<float>(100, 0.0f) }, { 0, vector<float>(100, 0.0f) } });
                        for (uint32_t i = range.first; i < range.second; ++i) {
                            // !could result in some groups never getting sampled!
                            if (sampleDist(rd) < sampleRate) {
                                auto grp = id_to_group[i];
                                auto& [offset, coefs] = groupPlanes[grp];
                                auto& pt = points[i];
                                auto& [agg1, agg2] = center_agg[grp];
                                auto& aggToUse = dot(coefs.data(), pt) >= offset ? agg1 : agg2;
                                aggToUse.first++;
                                plusEq(aggToUse.second.data(), pt);
                            }
                        }
                    });
                }
                for (auto& thread: threads) { thread.join(); }
                stage[3] += duration_cast<milliseconds>(hclock::now() - s3).count();


                // aggregate local results for split plane assignment
                auto s4 = hclock::now();
                vector<pair<centroid_agg, centroid_agg>> globalCenterAggs(numCurrGroups, { { 0, vector<float>(100, 0.0f) }, { 0, vector<float>(100, 0.0f) } });
                for (auto& local : localGroupCenterAggs) {
                    for (uint32_t g = 0; g < numCurrGroups; ++g) {
                        auto& ca = local[g];
                        auto& [c1, c2] = globalCenterAggs[g];
                        c1.first += ca.first.first;
                        c2.first += ca.second.first;
                        for (uint32_t j = 0; j < dims; ++j) { c1.second[j] += ca.first.second[j]; }
                        for (uint32_t j = 0; j < dims; ++j) { c2.second[j] += ca.second.second[j]; }
                    }
                }
                stage[4] += duration_cast<milliseconds>(hclock::now() - s4).count();

                auto s5 = hclock::now();
                for (uint32_t g = 0; g < numCurrGroups; ++g) {
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
                };
                stage[5] += duration_cast<milliseconds>(hclock::now() - s5).count();
            }

            // recompute groups based on plane
            auto s6 = hclock::now();
            vector<std::thread> threads;
            for (uint32_t t = 0; t < numThreads; ++t) {
                threads.emplace_back([&, t]() {
                    auto range = ranges[t];
                    for (uint32_t i = range.first; i < range.second; ++i) {
                        uint32_t grp = id_to_group[i];
                        auto& [offset, coefs] = groupPlanes[grp];
                        id_to_group[i] = dot(coefs.data(), points[i]) >= offset ? grp : grp + numCurrGroups;
                    }
                });
            }
            for (auto& thread: threads) { thread.join(); }
            stage[6] += duration_cast<milliseconds>(hclock::now() - s6).count();

            numCurrGroups = numNextGroups;
            numNextGroups = 2 * numCurrGroups;
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
        for (uint32_t g = 0; g < numCurrGroups; ++g) {
            auto& group = globalGrpToIds[g];
            if (group.empty()) {
                emptyGroups++;
            } else if (group.size() < 2'000) {
                groups.push_back(group);
            } else {
                groupsSkipped++;
                totalIdsSkipped+=group.size();
            }
        }
        std::cout << "num groups: " << groups.size() << "\n";
        std::cout << "num skipped groups: " << groupsSkipped << ", total size: " << totalIdsSkipped
            << ", avg size: " << static_cast<float>(totalIdsSkipped) / groupsSkipped << ", empty: " << emptyGroups << "\n";
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
            auto processDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();
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
