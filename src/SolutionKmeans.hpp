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
#include "tsl/robin_set.h"


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

    static pair<Vec, Vec> kmeansStartVecs(vector<uint32_t>& sampleRange, Range& range, float points[][112], vector<uint32_t>& indices) {
        uint32_t rangeSize = range.second - range.first;
        uint32_t sampleSizeforStarts = pow(log10(rangeSize), 2.5); // 129 samples for 10m bucket, 16 samples for bucket of 1220
        uint32_t sampleSize = std::min(static_cast<uint32_t>(sampleRange.size()), sampleSizeforStarts);

        auto sampleRangeStart = sampleRange.begin();
        auto sampleRangeEnd = sampleRange.begin() + sampleSize;

        float maxDist = std::numeric_limits<float>::min();
        float* pii = nullptr;
        float* pjj = nullptr;
        for (auto i = sampleRangeStart; i < sampleRangeEnd - 1; ++i) {
            for (auto j = i + 1; j < sampleRangeEnd; ++j) {
                float* pi = points[indices[*i]];
                float* pj = points[indices[*j]];
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

    static pair<Vec, Vec> kmeansStartVecs(Range& range, float points[][112], vector<uint32_t>& indices) {
        uint32_t rangeSize = range.second - range.first;
        uint32_t sampleSize = pow(log10(rangeSize), 2.5); // 129 samples for 10m bucket, 16 samples for bucket of 1220
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
    static void splitKmeansBinaryProcess(Range range,
                                     uint32_t knnIterations,
                                     uint32_t maxGroupSize,
                                     float points[][112],
                                     float pointsCopy[][112],
                                     vector<uint32_t>& indices,
                                     vector<KnnSetScannableSimd>& idToKnn,
                                     vector<float>& bounds
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            auto startProcess = hclock::now();
            addCandidatesLessThan(points, pointsCopy, indices, range, bounds, idToKnn);
            processTime += duration_cast<milliseconds>(hclock::now() - startProcess).count();
        } else if (rangeSize < 3'000) { // last two splits single threaded in hope of maintain cache locality
            begin_kmeans_small:

            auto [center1, center2] = kmeansStartVecs(range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());

                using centroid_agg = pair<uint32_t, vector<double>>;
                centroid_agg c1 = make_pair(0, vector<double>(100, 0.0));
                centroid_agg c2 = make_pair(0, vector<double>(100, 0.0));

                for (uint32_t i = range.first; i < range.second; ++i) {
                    auto& pt = points[indices[i]];
                    centroid_agg& ca = dot(coefs.data(), pt) >= offset ? c1 : c2;
                    ca.first++;
                    for (uint32_t j = 0; j < dims; ++j) { ca.second[j] += pt[j]; }
                }

                if (c1.first == 0 || c2.first == 0) {
                    goto begin_kmeans_small;
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

            auto indicesBegin = indices.begin() + range.first;
            auto indicesEnd = indices.begin() + range.second;
            auto middleIt = std::stable_partition(indicesBegin, indicesEnd, [&](uint32_t id) {
                return dot(coefs.data(), points[id]) >= offset;
            });
            auto range1Size = middleIt - indicesBegin;
            auto range2Size = indicesEnd - middleIt;
            Range lo = {range.first, range.first + range1Size};
            Range hi = {range.first + range1Size , range.second};

            if (range1Size == 0 || range2Size == 0) {
                goto begin_kmeans_small;
            }

            splitKmeansBinaryProcess(lo, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn, bounds);
            splitKmeansBinaryProcess(hi, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn, bounds);
        } else {
            begin_kmeans:

            float percSample = calcSamplePercent(range.first, range.second);
            auto sampleRange = getSampleFromPercent(percSample, range.first, range.second);

//            std::cout << "rangeSize: " << rangeSize << ", sampleSize: " << sampleRange.size() << "\n";
            auto [center1, center2] = kmeansStartVecs(sampleRange, range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());
                // dot(x, coefs) >= offset means nearer to center1


                using centroid_agg = pair<uint32_t, vector<double>>;
                tbb::combinable<pair<centroid_agg, centroid_agg>> agg(make_pair(make_pair(0, vector<double>(100, 0.0f)), make_pair(0, vector<double>(100, 0.0f))));
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, sampleRange.size()),
                    [&](oneapi::tbb::blocked_range<size_t> r) {
                        auto& [agg1, agg2] = agg.local();
                        for (uint32_t i = r.begin(); i < r.end(); ++i) {
                            auto& pt = points[indices[sampleRange[i]]];
                            auto& aggToUse = dot(coefs.data(), pt) >= offset ? agg1 : agg2;
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
                        auto& group = dot(coefs.data(), pt) >= offset ? g1 : g2;
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
                [&]{ splitKmeansBinaryProcess(subRange1, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn, bounds); },
                [&]{ splitKmeansBinaryProcess(subRange2, knnIterations, maxGroupSize, points, pointsCopy,indices, idToKnn, bounds); }
            );
        }
    }




//    static uint64_t topUp(float points[][112], vector<KnnSetScannable>& idToKnn) {
//        auto startTopup = hclock::now();
//        uint32_t numPoints = idToKnn.size();
//
//        std::atomic<uint64_t> nodesUpdated = 0;
//        std::atomic<uint64_t> nodesAdded = 0;
//
//        vector<vector<pair<uint32_t, bool>>> knnIds(numPoints);
//        tbb::parallel_for(
//            tbb::blocked_range<size_t>(0, numPoints),
//            [&](oneapi::tbb::blocked_range<size_t> r) {
//                for (auto id = r.begin(); id < r.end(); ++id) {
//                    vector<pair<uint32_t, bool>> ids;
//                    ids.reserve(100);
//                    for (auto &[dist, id2, isNew] : idToKnn[id].queue) {
//                        ids.emplace_back(id2, isNew);
//                        isNew = false;
//                    }
//                    std::sort(ids.begin(), ids.end());
//                    knnIds[id] = std::move(ids);
//                }
//            }
//        );
//
//        std::cout << "top copy idKnnSet time: " << duration_cast<milliseconds>(hclock::now() - startTopup).count() << "\n";
//
//        std::atomic<uint32_t> count = 0;
//        tbb::parallel_for(
//            tbb::blocked_range<size_t>(0, numPoints),
//            [&](oneapi::tbb::blocked_range<size_t> r) {
//                tsl::robin_set<uint32_t> candidates;
//                for (auto id1 = r.begin(); id1 < r.end(); ++id1) {
//                    uint32_t added = 0;
//                    auto& knn = knnIds[id1];
//                    auto& knnSet = idToKnn[id1];
//                    for (auto& [id2, isNew2] : knnIds[id1]) {
//                        if (isNew2) {
//                            // if id2 is new, need to add all neighbors
//                            for (auto& [id3, isNew3]: knnIds[id2]) { candidates.insert(id3); }
//                        } else {
//                            // if id2 is not new, only need to add its new neighbors
//                            for (auto& [id3, isNew3]: knnIds[id2]) {
//                                if (isNew3) { candidates.insert(id3); }
//                            }
//                        }
//                    }
//
//                    // remove current ids and self id
////                    for (auto& id2 : knnIds[id1]) { candidates.erase(id2); }
//                    candidates.erase(id1);
//
//                    for (auto& id3 : candidates) {
//                        float dist = distance(points[id3], points[id1]);
//                        if (knnSet.addCandidate(id3, dist)) {
//                            added++;
//                        }
//                    }
//                    candidates.clear();
//
//                    auto currCount = count++;
//                    if (currCount % 10'000 == 0) {
//                        auto topupTime = duration_cast<milliseconds>(hclock::now() - startTopup).count();
//                        std::cout << "topped up: " << currCount << ", timing topping up:" << topupTime << "\n";
//                    }
//                    if (added > 0) {
//                        nodesUpdated++;
//                        nodesAdded += added;
//                    }
//                }
//            }
//        );
//
//        std::cout << "topUp nodes added: " << nodesAdded << "\n";
//        std::cout << "topUp nodes changed: " << nodesUpdated << "\n";
//        return nodesUpdated.load();
//    }

    static uint64_t topUpSingleOrdered(float points[][112], vector<KnnSetScannableSimd>& idToKnn, vector<float>& bounds, long timeBoundMs, auto startTime) {
        auto startTopup = hclock::now();
        uint32_t numPoints = idToKnn.size();

        std::atomic<uint64_t> nodesUpdated = 0;
        std::atomic<uint64_t> nodesAdded = 0;

        vector<vector<uint32_t>> knnIds(numPoints);
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, numPoints),
                [&](oneapi::tbb::blocked_range<size_t> r) {
                    for (auto id = r.begin(); id < r.end(); ++id) {
                        auto& knn = idToKnn[id];
                        vector<pair<float, uint32_t>> distIds;
                        distIds.reserve(knn.size);
                        for (uint32_t i = 0; i < knn.size; ++i) {
                            distIds.emplace_back(knn.dists[i], knn.current_ids[i]);
                        }
                        std::sort(distIds.begin(), distIds.end());
                        vector<uint32_t> ids;
                        ids.reserve(knn.size);
                        for (auto& [dist, id2] : distIds) {
                            ids.push_back(id2);
                        }
                        knnIds[id] = std::move(ids);
                    }
                }
        );

        std::cout << "top copy idKnnSet time: " << duration_cast<milliseconds>(hclock::now() - startTopup).count() << "\n";

        std::atomic<uint32_t> count = 0;
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, numPoints),
                [&](oneapi::tbb::blocked_range<size_t> r) {
                    if (duration_cast<milliseconds>(hclock::now() - startTime).count() >= timeBoundMs) {
                        return;
                    }

                    tsl::robin_set<uint32_t> candidates;
                    for (auto id1 = r.begin(); id1 < r.end(); ++id1) {
                        uint32_t added = 0;
                        auto& knn = knnIds[id1];
                        auto& knn1 = idToKnn[id1];
                        auto& ki1 = knnIds[id1];
                        for (uint32_t i2 = 50; i2 < ki1.size(); ++i2) {
                            auto& ki2 = knnIds[ki1[i2]];
                            for (uint32_t i3 = 0; i3 < ki2.size() && i3 < 100; ++i3) {
                                auto id3 = ki2[i3];
                                candidates.insert(id3);
                            }
                        }

                        // remove current ids and self id
//                    for (auto& id2 : knnIds[id1]) { candidates.erase(id2); }
                        candidates.erase(id1);

                        for (auto& id3 : candidates) {
                            float dist = distance(points[id3], points[id1]);
                            float& bound = bounds[id1];
                            if (dist < bound) {
                                if (knn1.addCandidateLessThan(bound, id3, dist)) {
                                    added++;
                                }
                            }
                        }
                        candidates.clear();

                        auto currCount = count++;
                        if (currCount % 10'000 == 0) {
                            auto topupTime = duration_cast<milliseconds>(hclock::now() - startTopup).count();
                            std::cout << "topped up: " << currCount << ", timing topping up:" << topupTime << ", nodes added: " << nodesAdded << ", nodes changed: " << nodesUpdated << "\n";
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


    static uint64_t topUpSingle(float points[][112], vector<KnnSetScannableSimd>& idToKnn, vector<float>& bounds, long timeBoundMs, auto startTime) {
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
                    if (duration_cast<milliseconds>(hclock::now() - startTime).count() >= timeBoundMs) {
                        return;
                    }

                    tsl::robin_set<uint32_t> candidates;
                    for (auto id1 = r.begin(); id1 < r.end(); ++id1) {
                        uint32_t added = 0;
                        auto& knn = knnIds[id1];
                        auto& knn1 = idToKnn[id1];
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
                            float& bound = bounds[id1];
                            if (dist < bound) {
                                if (knn1.addCandidateLessThan(bound, id3, dist)) {
                                    added++;
                                }
                            }
                        }
                        candidates.clear();

                        auto currCount = count++;
                        if (currCount % 10'000 == 0) {
                            auto topupTime = duration_cast<milliseconds>(hclock::now() - startTopup).count();
                            std::cout << "topped up: " << currCount << ", timing topping up:" << topupTime << ", nodes added: " << nodesAdded << ", nodes changed: " << nodesUpdated << "\n";
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


    static void constructResult(float points[][112], uint32_t numPoints, vector<vector<uint32_t>>& result) {

        bool localRun = getenv("LOCAL_RUN");
        auto numThreads = std::thread::hardware_concurrency();
        long timeBoundMs = (localRun || numPoints == 10'000)  ? 20'000 : 1'210'000;
        long topUp = (localRun || numPoints == 10'000)  ? 20'000 : 500'000;
        long timeBoundTopUp = timeBoundMs + topUp;

        float (*pointsCopy)[112] = static_cast<float(*)[112]>(aligned_alloc(64, numPoints * 112 * sizeof(float)));

        std::cout << "start run with time bound: " << timeBoundMs << ", topUp bound: " << timeBoundTopUp << "\n";

        auto startTime = hclock::now();
        vector<KnnSetScannableSimd> idToKnn(numPoints);

        // rewrite point data in adjacent memory and sort in a group order
        std::vector<uint32_t> indices(numPoints);
        vector<float> bounds(numPoints, std::numeric_limits<float>::max());


        uint32_t iteration = 0;
        while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundMs) {
            std::cout << "Iteration: " << iteration << '\n';

            std::iota(indices.begin(), indices.end(), 0);
            auto startGroupProcess = hclock::now();
            splitKmeansBinaryProcess({0, numPoints}, 1, 400, points, pointsCopy, indices, idToKnn, bounds);

            auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroupProcess).count();
            std::cout << " group/process time: " << groupDuration << '\n';
            groupProcessTime += groupDuration;
            uint64_t avgProcessTime = processTime / numThreads;
            std::cout << " avg group time: " << groupDuration - avgProcessTime << '\n';
            std::cout << " avg process time: " << avgProcessTime << '\n';
            processTime = 0;

            iteration++;
        }

        topUpSingle(points, idToKnn, bounds, timeBoundTopUp, startTime);

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
