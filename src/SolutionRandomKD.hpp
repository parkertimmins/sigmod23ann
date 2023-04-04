#ifndef SIGMOD23ANN_SOLUTIONRANDOMKD_HPP
#define SIGMOD23ANN_SOLUTIONRANDOMKD_HPP

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

struct SolutionRandomKD {

    static inline std::atomic<uint64_t> groupProcessTime = 0;


    static uint32_t calcSampleSize(uint32_t min, uint32_t max) {
        uint32_t rangeSize = max - min;
        return rangeSize / 3;
        //return pow(log(rangeSize) / log(30), 7);
    }


    static vector<uint32_t> getSample(uint32_t min, uint32_t max) {
        uint32_t sampleSize = calcSampleSize(min, max);
        vector<uint32_t> sample;
        sample.reserve(sampleSize);
        std::uniform_int_distribution<uint32_t> distribution(min, max-1);
        while (sample.size() < sampleSize) {
            sample.push_back(distribution(rd));
        }
        return sample;
    }


    static pair<Vec, Vec> kmeansStartVecs(Range& range, float points[][104], vector<uint32_t>& indices) {
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

    static vector<double> getMeans(float points[][104], vector<uint32_t> indices, Range range) {
        uint32_t rangeSize = range.second - range.first;
        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, rangeSize),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                auto& sumsLocal = sums.local();
                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                    auto& pt = points[indices[i]];
                    for (uint32_t j = 0; j < dims; ++j) { sumsLocal[j] += pt[j]; }
                }
            }
        );
        auto sumsGlobal = sums.combine([](const vector<double>& x, const vector<double>& y) {
            vector<double> res(100, 0.0f);
            for (uint32_t j = 0; j < dims; ++j) { res[j]  = x[j] + y[j]; }
            return res;
        });

        for (auto& v : sumsGlobal) {
            v /= rangeSize;
        }
        return sumsGlobal;
    }

    static vector<double> getVariances(vector<double> means, float points[][104], vector<uint32_t> indices, Range range) {
        uint32_t rangeSize = range.second - range.first;
        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, rangeSize),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                auto& sumsLocal = sums.local();
                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                    auto& pt = points[indices[i]];
                    for (uint32_t j = 0; j < dims; ++j) {
                        auto diff = pt[j] - means[j];
                        sumsLocal[j] += diff * diff;
                    }
                }
            }
        );
        auto sumsGlobal = sums.combine([](const vector<double>& x, const vector<double>& y) {
            vector<double> res(100, 0.0f);
            for (uint32_t j = 0; j < dims; ++j) { res[j]  = x[j] + y[j]; }
            return res;
        });

        for (auto& v : sumsGlobal) {
            v /= (rangeSize - 1);
        }
        return sumsGlobal;
    }

//    static float getMedian(uint32_t dimIdx, float points[][104], vector<uint32_t> indices, Range range) {
//        uint32_t rangeSize = range.second - range.first;
//        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
//        tbb::parallel_for(
//            tbb::blocked_range<size_t>(0, rangeSize),
//            [&](oneapi::tbb::blocked_range<size_t> r) {
//                auto& sumsLocal = sums.local();
//                for (uint32_t i = r.begin(); i < r.end(); ++i) {
//                    auto& pt = points[indices[i]];
//                    for (uint32_t j = 0; j < dims; ++j) { sumsLocal[j] += pt[j]; }
//                }
//            }
//        );
//        auto sumsGlobal = sums.combine([](const vector<double>& x, const vector<double>& y) {
//            vector<double> res(100, 0.0f);
//            for (uint32_t j = 0; j < dims; ++j) { res[j]  = x[j] + y[j]; }
//            return res;
//        });
//
//        for (auto& v : sumsGlobal) {
//            v /= rangeSize;
//        }
//        return sumsGlobal;
//    }

    static uint32_t pickIndex(Range range, float points[][104], vector<uint32_t>& indices) {
        auto means = getMeans(points, indices, range);
        auto variances = getVariances(means, points, indices, range);

        // pick dimension
        vector<pair<double, uint32_t>> varianceIndices;
        varianceIndices.reserve(100);
        for (uint32_t j = 0; j < dims; ++j) {
            varianceIndices.emplace_back(variances[j], j);
        }
        std::sort(varianceIndices.begin(), varianceIndices.end(), std::greater{});
        uint32_t indexSampleRange = 30;
        std::uniform_int_distribution<uint32_t> distribution(0, indexSampleRange);
        auto [var, idx] = varianceIndices[distribution(rd)];
        return idx;
    }

    static uint32_t pickRandomIndex() {
        std::uniform_int_distribution<uint32_t> distribution(0, 99);
        return distribution(rd);
    }

    static void split(Range range, uint32_t maxGroupSize, float points[][104], vector<uint32_t>& indices, vector<KnnSetScannable>& idToKnn) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            addCandidates(points, indices, range, idToKnn);
        } else {
//            uint32_t idx = pickIndex(range, points, indices);
            uint32_t idx = pickRandomIndex();

            // find median
            auto sampleRange = getSample(range.first, range.second);
            vector<float> sample;
            sample.reserve(sampleRange.size());
            for (auto& i : sampleRange) {
                sample.push_back(points[indices[i]][idx]);
            }

//            vector<float> sample;
//            sample.reserve(rangeSize);
//            for (uint32_t i = range.first; i < range.second; ++i) {
//                float* pt = points[indices[i]];
//                sample.push_back(pt[idx]);
//            }

            std::sort(sample.begin(), sample.end());
            auto median = sample[sample.size() / 2];
            auto splitValue = median; //means[idx];

            // compute final groups
            using groups = pair<vector<uint32_t>, vector<uint32_t>>;
            tbb::combinable<groups> groupsAgg(make_pair<>(vector<uint32_t>(), vector<uint32_t>()));
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(range.first, range.second),
                [&](tbb::blocked_range<uint32_t> r) {
                    auto& [g1, g2] = groupsAgg.local();
                    for (uint32_t i = r.begin(); i < r.end(); ++i) {
                        auto id = indices[i];
                        auto& pt = points[id];
                        auto& group = pt[idx] < splitValue ? g1 : g2;
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
                [&]{ split(subRange1, maxGroupSize, points, indices, idToKnn); },
                [&]{ split(subRange2, maxGroupSize, points, indices, idToKnn); }
            );
        }
    }

    static void constructResult(float points[][104], uint32_t numPoints, vector<vector<uint32_t>>& result) {

        long timeBoundsMs = (getenv("LOCAL_RUN") || numPoints == 10'000)  ? 20'000 : 1'650'000;

    #ifdef PRINT_OUTPUT
        std::cout << "start run with time bound: " << timeBoundsMs << '\n';
    #endif
        auto startTime = hclock::now();
        vector<KnnSetScannable> idToKnn(numPoints);

        // rewrite point data in adjacent memory and sort in a group order
        std::vector<uint32_t> indices(numPoints);

        uint32_t iteration = 0;
//        while (iteration < 5) {
        while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
            std::cout << "Iteration: " << iteration << '\n';

            std::iota(indices.begin(), indices.end(), 0);
            auto startGroupProcess = hclock::now();
            split({0, numPoints}, 400, points, indices, idToKnn);

            auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroupProcess).count();
            std::cout << " group/process time: " << groupDuration << '\n';
            groupProcessTime += groupDuration;

            iteration++;
        }

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
#endif //SIGMOD23ANN_SOLUTIONRANDOMKD_HPP
