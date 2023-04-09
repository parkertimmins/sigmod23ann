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
#include <cmath>
#include "tsl/robin_map.h"


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
                                     vector<KnnSetScannable>& idToKnn
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            auto startProcess = hclock::now();
            addCandidatesCopy(points, pointsCopy, indices, range, idToKnn);
            processTime += duration_cast<milliseconds>(hclock::now() - startProcess).count();
        } else if (rangeSize < 3'000) { // last two splits single threaded in hope of maintain cache locality
            begin_kmeans_small:

            float percSample = calcSamplePercent(range.first, range.second);
            auto sampleRange = getSampleFromPercent(percSample, range.first, range.second);
            auto [center1, center2] = kmeansStartVecs(sampleRange, range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());

                using centroid_agg = pair<uint32_t, vector<float>>;
                centroid_agg c1 = make_pair(0, vector<float>(100, 0.0));
                centroid_agg c2 = make_pair(0, vector<float>(100, 0.0));

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

            splitKmeansBinaryProcess(lo, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn);
            splitKmeansBinaryProcess(hi, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn);
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


                using centroid_agg = pair<uint32_t, vector<float>>;
                tbb::combinable<pair<centroid_agg, centroid_agg>> agg(make_pair(make_pair(0, vector<float>(100, 0.0f)), make_pair(0, vector<float>(100, 0.0f))));
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
                    centroid_agg c1{0, vector<float>(100, 0.0f)};
                    centroid_agg c2{0, vector<float>(100, 0.0f)};
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
                [&]{ splitKmeansBinaryProcess(subRange1, knnIterations, maxGroupSize, points, pointsCopy, indices, idToKnn); },
                [&]{ splitKmeansBinaryProcess(subRange2, knnIterations, maxGroupSize, points, pointsCopy,indices, idToKnn); }
            );
        }
    }

    inline static vector<long> depthTimes;

    static vector<pair<float, uint32_t>> getSortedIndices(Vec& v) {
        vector<pair<float, uint32_t>> with_idx;
        for (uint32_t i = 0; i < 100; ++i) {
            with_idx.emplace_back(v[i], i);
        }
        std::sort(v.begin(), v.end(), std::greater{});
        return with_idx;
    }


    static vector<double> getMeanSample(float points[][112], vector<uint32_t>& indices, vector<uint32_t>& rangeSample) {
        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, rangeSample.size(), 1000),
            [&](oneapi::tbb::blocked_range<size_t> r) {
                auto& sumsLocal = sums.local();
                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                    auto& pt = points[indices[rangeSample[i]]];
                    for (uint32_t j = 0; j < dims; ++j) { sumsLocal[j] += pt[j]; }
                }
            }
        );
        auto sumsGlobal = sums.combine([](const vector<double>& x, const vector<double>& y) {
            vector<double> res(100, 0.0f);
            for (uint32_t j = 0; j < dims; ++j) { res[j]  = x[j] + y[j]; }
            return res;
        });

        auto size = rangeSample.size();
        for (auto& v : sumsGlobal) {
            v /= size;
        }
        return sumsGlobal;
    }

    static vector<double> getMean(float points[][112], vector<uint32_t>& indices, Range range) {
        uint32_t rangeSize = range.second - range.first;
        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, rangeSize, 1000),
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
    static uint32_t calcSampleSize(uint32_t min, uint32_t max) {
        uint32_t rangeSize = max - min;
//        return rangeSize / 3;
        return pow(log(rangeSize) / log(30), 7);
    }
    static vector<double> getVarianceSample(vector<double>& means, float points[][112], vector<uint32_t>& indices, vector<uint32_t>& rangeSample) {
        tbb::combinable<vector<double>> sums(vector<double>(100, 0.0f));
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, rangeSample.size(), 1000),
                [&](oneapi::tbb::blocked_range<size_t> r) {
                    auto& sumsLocal = sums.local();
                    for (uint32_t i = r.begin(); i < r.end(); ++i) {
                        auto& pt = points[indices[rangeSample[i]]];
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

        auto size = rangeSample.size();
        for (auto& v : sumsGlobal) {
            v /= (size - 1);
        }
        return sumsGlobal;
    }

    static uint32_t pickIndex(float points[][112], vector<uint32_t>& indices, vector<uint32_t>& rangeSample) {
        auto means = getMeanSample(points, indices, rangeSample);
        auto variances = getVarianceSample(means, points, indices, rangeSample);

        // pick dimension
        vector<pair<double, uint32_t>> varianceIndices;
        varianceIndices.reserve(100);
        for (uint32_t j = 0; j < dims; ++j) {
            varianceIndices.emplace_back(variances[j], j);
        }
        std::sort(varianceIndices.begin(), varianceIndices.end(), std::greater{});
        uint32_t indexSampleRange = 15;
        std::uniform_int_distribution<uint32_t> distribution(0, indexSampleRange);
        auto [var, idx] = varianceIndices[distribution(rd)];
        return idx;
    }


    // handle both point vector data and array data
    static void splitKmeansBinary(Range range,
                                     uint32_t knnIterations,
                                     uint32_t maxGroupSize,
                                     float points[][112],
                                     float pointsCopy[][112],
                                     float* pointsCol,
                                     vector<uint32_t>& indices,
                                     tbb::concurrent_vector<Range>& ranges,
                                     bool shouldSplit,
                                     uint32_t depth,
                                     uint32_t numPoints
    ) {
        uint32_t rangeSize = range.second - range.first;
        if (rangeSize < maxGroupSize) {
            ranges.push_back(range);
        } else if (rangeSize < 3'000) { // last two splits single threaded in hope of maintain cache locality
            auto startDepth = hclock::now();

            begin_kmeans_small:

            float percSample = calcSamplePercent(range.first, range.second);
            auto sampleRange = getSampleFromPercent(percSample, range.first, range.second);
            auto [center1, center2] = kmeansStartVecs(sampleRange, range, points, indices);

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {
                auto between = scalarMult(0.5, add(center1, center2));
                auto coefs = sub(center1, between);
                auto offset = dot(between.data(), coefs.data());

                using centroid_agg = pair<uint32_t, vector<float>>;
                centroid_agg c1 = make_pair(0, vector<float>(100, 0.0));
                centroid_agg c2 = make_pair(0, vector<float>(100, 0.0));

                for (uint32_t i = range.first; i < range.second; ++i) {
                    auto &pt = points[indices[i]];
                    centroid_agg &ca = dot(coefs.data(), pt) >= offset ? c1 : c2;
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
            Range hi = {range.first + range1Size, range.second};

            if (range1Size == 0 || range2Size == 0) {
                goto begin_kmeans_small;
            }

            auto depthDuration = duration_cast<milliseconds>(hclock::now() - startDepth).count();
            depthTimes[depth] += depthDuration;

            splitKmeansBinary(lo, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, false, depth + 1, numPoints);
            splitKmeansBinary(hi, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, false, depth + 1, numPoints);
        } else if (depth < 6) {
            auto startDepth = hclock::now();

            // get range sample
            auto sampleRange = getSample(range.first, range.second);
            uint32_t idx = pickIndex(points, indices, sampleRange);

            // get sample of column
            vector<float> sample;
            sample.reserve(sampleRange.size());
            for (auto& i : sampleRange) {
                auto id = indices[i];
                sample.push_back(pointsCol[idx * numPoints + id]);
            }

            // 40 - 60 quantiles?
            std::sort(sample.begin(), sample.end());
            auto median = sample[sample.size() / 2];
            auto splitValue = median;

            // compute final groups
            using groups = pair<vector<uint32_t>, vector<uint32_t>>;
            tbb::combinable<groups> groupsAgg(make_pair<>(vector<uint32_t>(), vector<uint32_t>()));
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(range.first, range.second),
                [&](tbb::blocked_range<uint32_t> r) {
                    auto& [g1, g2] = groupsAgg.local();
                    for (uint32_t i = r.begin(); i < r.end(); ++i) {
                        auto id = indices[i];
                        auto& group = pointsCol[idx * numPoints + id] < splitValue ? g1 : g2;
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

            auto depthDuration = duration_cast<milliseconds>(hclock::now() - startDepth).count();
            depthTimes[depth] += depthDuration;

            tbb::parallel_invoke(
            [&]{
                auto it1 = indices.data() + subRange1Start;
                std::memcpy(it1, group1.data(), group1.size() * sizeof(uint32_t));
                splitKmeansBinary(subRange1, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, shouldSplit, depth+1, numPoints);
            },
            [&]{
                auto it2 = indices.data() + subRange2Start;
                std::memcpy(it2, group2.data(), group2.size() * sizeof(uint32_t));
                splitKmeansBinary(subRange2, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, shouldSplit, depth+1, numPoints);
            });
        } else {
            auto startDepth = hclock::now();

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


                using centroid_agg = pair<uint32_t, vector<float>>;
                tbb::combinable<pair<centroid_agg, centroid_agg>> agg(make_pair(make_pair(0, vector<float>(100, 0.0f)), make_pair(0, vector<float>(100, 0.0f))));
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
                    centroid_agg c1{0, vector<float>(100, 0.0f)};
                    centroid_agg c2{0, vector<float>(100, 0.0f)};
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

            auto depthDuration = duration_cast<milliseconds>(hclock::now() - startDepth).count();
            depthTimes[depth] += depthDuration;

            tbb::parallel_invoke(
            [&]{
                auto it1 = indices.data() + subRange1Start;
                std::memcpy(it1, group1.data(), group1.size() * sizeof(uint32_t));
                splitKmeansBinary(subRange1, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, shouldSplit, depth+1, numPoints);
            },
            [&]{
                auto it2 = indices.data() + subRange2Start;
                std::memcpy(it2, group2.data(), group2.size() * sizeof(uint32_t));
                splitKmeansBinary(subRange2, knnIterations, maxGroupSize, points, pointsCopy, pointsCol, indices, ranges, shouldSplit, depth+1, numPoints);
            });
        }
    }

//
//    static uint64_t topUpSingle(float points[][112], vector<KnnSetScannableSimd>& idToKnn) {
//        auto startTopup = hclock::now();
//        uint32_t numPoints = idToKnn.size();
//
//        std::atomic<uint64_t> nodesUpdated = 0;
//        std::atomic<uint64_t> nodesAdded = 0;
//
//        vector<vector<uint32_t>> knnIds(numPoints);
//        tbb::parallel_for(
//            tbb::blocked_range<size_t>(0, numPoints),
//            [&](oneapi::tbb::blocked_range<size_t> r) {
//                for (auto id = r.begin(); id < r.end(); ++id) {
//                    vector<uint32_t> ids;
//                    ids.reserve(100);
//                    for (auto& id2 : idToKnn[id].current_ids) {
//                        ids.push_back(id2);
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
//                    for (auto& id2 : knnIds[id1]) {
//                        for (auto& id3 : knnIds[id2]) {
//                            candidates.insert(id3);
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


    static tsl::robin_map<uint32_t, pair<uint32_t, vector<uint32_t>>> getPerGroupsSample(uint32_t numPoints,
                                                                                         vector<uint32_t>& id_to_group) {
        static uint32_t resSize = 2;
        tsl::robin_map<uint32_t, pair<uint32_t, vector<uint32_t>>> reservoirs;
        for (uint32_t i = 0; i < numPoints; ++i) {
            uint32_t grp = id_to_group[i];
            if (reservoirs.find(grp) == reservoirs.end()) {
                reservoirs[grp] = {0, vector<uint32_t>()};
            }
            auto &[numSeen, reservoir] = reservoirs[grp];
            if (reservoir.size() < resSize) {
                reservoir.push_back(i);
            } else {
                std::uniform_int_distribution<uint32_t> distribution(0, numSeen);
                uint32_t j = distribution(rd);
                if (j < resSize) {
                    reservoir[j] = i;
                }
            }
            numSeen++;
        }
        return reservoirs;
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

    inline static long stage1 = 0;
    inline static long stage2 = 0;
    inline static long stage3 = 0;
    inline static long stage4 = 0;
    inline static long stage5 = 0;
    inline static long stage6 = 0;
    inline static long stage7 = 0;
    inline static long stage8 = 0;
    inline static long stage9 = 0;

    // handle both point vector data and array data
    static void splitKmeansNonRec(
            uint32_t numPoints,
            uint32_t knnIterations,
            uint32_t maxGroupSize,
            float points[][112],
            vector<KnnSetScannable>& idToKnn) {


        stage1 = 0;
        stage2 = 0;
        stage3 = 0;
        stage4 = 0;
        stage5 = 0;
        stage6 = 0;
        stage7 = 0;
        stage8 = 0;
        stage9 = 0;


        vector<uint32_t> id_to_group(numPoints, 1);
        uint32_t depth = requiredHashFuncs(numPoints, maxGroupSize);
        uint32_t d = 0;
        while (d++ < depth) {
            auto s1 = hclock::now();
            auto perGroupSamples = getPerGroupsSample(numPoints, id_to_group);
            stage1 += duration_cast<milliseconds>(hclock::now() - s1).count();

            auto s2 = hclock::now();
            // build centers;
            tsl::robin_map<uint32_t, pair<Vec, Vec>> groupCenters;
            tsl::robin_map<uint32_t, pair<float, Vec>> groupPlane;
            for (auto& [grp, countSample] : perGroupSamples) {
                auto& [count, sample] = countSample;
                Vec c1(100);
                Vec c2(100);
                for (uint32_t j = 0; j < 100; ++j) { c1[j] = points[sample[0]][j]; }
                for (uint32_t j = 0; j < 100; ++j) { c2[j] = points[sample[1]][j]; }
                groupCenters[grp] = { c1, c2 };

                auto between = scalarMult(0.5, add(c1, c2));
                auto coefs = sub(c1, between);
                auto offset = dot(between.data(), coefs.data());
                groupPlane[grp] = { offset, coefs };
            };
            stage2 += duration_cast<milliseconds>(hclock::now() - s2).count();

            for (uint32_t iteration = 0; iteration < knnIterations; ++iteration) {

                auto s3 = hclock::now();
                using centroid_agg = pair<uint32_t, vector<float>>;
                using group_center_agg = tsl::robin_map<uint32_t, pair<centroid_agg, centroid_agg>>;
                tbb::combinable<group_center_agg> agg;
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, numPoints),
                    [&](oneapi::tbb::blocked_range<size_t> r) {
                        auto& center_agg = agg.local();
                        for (uint32_t i = r.begin(); i < r.end(); ++i) {
                            auto grp = id_to_group[i];
                            auto& [offset, coefs] = groupPlane[grp];

                            if (center_agg.find(grp) == center_agg.end()) {
                                center_agg[grp] = { { 0, vector<float>(100, 0.0f) }, { 0, vector<float>(100, 0.0f) } };
                            }

                            auto& pt = points[i];
                            auto& [agg1, agg2] = center_agg[grp];
                            auto& aggToUse = dot(coefs.data(), points[i]) >= offset ? agg1 : agg2;
                            aggToUse.first++;
                            for (uint32_t j = 0; j < dims; ++j) { aggToUse.second[j] += pt[j]; }
                        }
                    }
                );
                stage3 += duration_cast<milliseconds>(hclock::now() - s3).count();

                auto s4 = hclock::now();
                auto per_group_center_aggs = agg.combine([](const group_center_agg& x, const group_center_agg& y) {
                    group_center_agg res;
                    res.insert(x.begin(), x.end());
                    for (auto& [grp, ca] : y) {
                        if (res.find(grp) == res.end()) {
                            res[grp] = ca;
                        } else {
                            auto& [c1, c2] = res[grp];
                            c1.first += ca.first.first;
                            c2.first += ca.second.first;
                            for (uint32_t j = 0; j < dims; ++j) {
                                c1.second[j] += ca.first.second[j];
                                c2.second[j] += ca.second.second[j];
                            }
                        }
                    }
                    return res;
                });
                stage4 += duration_cast<milliseconds>(hclock::now() - s4).count();

                auto s5 = hclock::now();
                for (auto& [grp, center_aggs] : per_group_center_aggs)  {
                    auto& [center1, center2] = groupCenters[grp];
                    auto& [c1_agg, c2_agg] = center_aggs;
                    for (uint32_t i = 0; i < dims; ++i) {
                        center1[i] = c1_agg.second[i] / c1_agg.first;
                        center2[i] = c2_agg.second[i] / c2_agg.first;
                    }
                    auto between = scalarMult(0.5, add(center1, center2));
                    auto coefs = sub(center1, between);
                    auto offset = dot(between.data(), coefs.data());
                    groupPlane[grp] = { offset, coefs };
                };
                stage5 += duration_cast<milliseconds>(hclock::now() - s5).count();
            }

            auto s6 = hclock::now();
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(0, numPoints),
                [&](tbb::blocked_range<uint32_t>& r) {
                    for (uint32_t i = r.begin(); i < r.end(); ++i) {
                        uint32_t grp = id_to_group[i];
                        auto& [offset, coefs] = groupPlane[grp];
                        id_to_group[i] = dot(coefs.data(), points[i]) >= offset ? 2 * grp : 2 * grp + 1;
                    }
                }
            );
            stage6 += duration_cast<milliseconds>(hclock::now() - s6).count();
        }

        auto s7 = hclock::now();
        using grp_to_group = tsl::robin_map<uint32_t, vector<uint32_t>>;
        tbb::combinable<grp_to_group> final_groups;
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, numPoints),
            [&](tbb::blocked_range<uint32_t>& r) {
                auto& local_final_groups = final_groups.local();
                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                    uint32_t grp = id_to_group[i];
                    if (local_final_groups.find(grp) == local_final_groups.end()) {
                        local_final_groups[grp] = vector<uint32_t>();
                    }
                    local_final_groups[grp].push_back(i);
                }
            }
        );
        auto final = final_groups.combine([](const grp_to_group& x, const grp_to_group& y) {
            grp_to_group res;
            for (auto& [grp, group] : x) { res[grp] = group; }
            for (auto& [grp, group_y] : y) {
                if (res.find(grp) == res.end()) {
                    res[grp] = group_y;
                } else {
                    auto& group_x = res[grp];
                    group_x.insert(group_x.end(), group_y.begin(), group_y.end());
                }
            }
            return res;
        });
        stage7 += duration_cast<milliseconds>(hclock::now() - s7).count();

        auto s8 = hclock::now();
        vector<vector<uint32_t>> groups;
        groups.reserve(final.size());
        for (auto& [id, group] : final) {
            if (group.size() < 3 * maxGroupSize) {
                groups.push_back(group);
            } else {
                std::cout << "too large: " << group.size() << "\n";
            }
        }
        std::cout << "num groups: " << groups.size() << "\n";
        final.clear();
        stage8 += duration_cast<milliseconds>(hclock::now() - s8).count();

        auto s9 = hclock::now();
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, groups.size()),
            [&](tbb::blocked_range<uint32_t> r) {
                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                    addCandidatesGroup(points, groups[i], idToKnn);
                }
            }
        );
        stage9 += duration_cast<milliseconds>(hclock::now() - s9).count();
    }

    static void constructResult(float points[][112], uint32_t numPoints, vector<vector<uint32_t>>& result) {

        bool localRun = getenv("LOCAL_RUN");
        auto numThreads = std::thread::hardware_concurrency();
        long timeBoundsMs = (localRun || numPoints == 10'000)  ? 20'000 : 1'650'000;


        float (*pointsCopy)[112] = static_cast<float(*)[112]>(aligned_alloc(64, numPoints * 112 * sizeof(float)));
        float* pointsCol = static_cast<float*>(aligned_alloc(64, numPoints * 100 * sizeof(float)));

        for (uint32_t r = 0; r < numPoints; ++r) {
            for (uint32_t c = 0; c < 100; ++c) {
                pointsCol[c * numPoints + r] = points[r][c];
            }
        }

        std::cout << "start run with time bound: " << timeBoundsMs << '\n';

        auto startTime = hclock::now();
        vector<float> bounds(numPoints, std::numeric_limits<float>::max());

        vector<KnnSetScannableSimd> idToKnn(numPoints);

        // rewrite point data in adjacent memory and sort in a group order
        std::vector<uint32_t> indices(numPoints);

        tbb::concurrent_vector<Range> ranges;
        uint32_t iteration = 0;
//        while (iteration < 10) {
        while (duration_cast<milliseconds>(hclock::now() - startTime).count() < timeBoundsMs) {
            std::cout << "Iteration: " << iteration << '\n';

            std::iota(indices.begin(), indices.end(), 0);

            depthTimes.clear(); depthTimes.resize(50, 0); // should never need depth 100!

            auto startGroup = hclock::now();
            splitKmeansBinary({0, numPoints}, 1, 400, points, pointsCopy, pointsCol, indices, ranges, true, 0, numPoints);
            auto groupDuration = duration_cast<milliseconds>(hclock::now() - startGroup).count();

            std::cout << "num ranges: " << ranges.size() << "\n";
            auto startProcess = hclock::now();
            Task<Range, tbb::concurrent_vector<Range>> tasks(ranges);
            vector<std::thread> threads;
            for (uint32_t t = 0; t < numThreads; ++t) {
                threads.emplace_back([&, t]() {
                    auto task = tasks.getTask();
                    while (task) {
                        auto range = *task;
                        addCandidatesLessThan(points, pointsCopy, indices, range, bounds, idToKnn);
                        task = tasks.getTask();
                    }
                });
            }
            for (auto& thread: threads) { thread.join(); }
            auto processDuration = duration_cast<milliseconds>(hclock::now() - startProcess).count();

            std::cout << " group time: " << groupDuration << '\n';
            std::cout << " process time: " << processDuration << '\n';
            processTime = 0;

            iteration++;

            for (uint32_t d = 0; d < depthTimes.size(); ++d) {
                if (depthTimes[d] > 0) {
                    std::cout << "depth " << d << " time: " << depthTimes[d] << "\n";
                }
            }
            ranges.clear();
        }

//        topUpSingle(points, idToKnn);

        for (uint32_t id = 0; id < numPoints; ++id) {
            result[id] = idToKnn[id].finalize();
        }

//        for (uint32_t i = 0; i < numPoints; ++i) {
//            float lb = bounds[i];
//            auto& knn = idToKnn[i];
//            if (knn.size < 100) {
//                std::cout << "i: " << i << ", lower bound: " << lb << ", size: " << knn.size << "\n";
//            }
//        }

        auto sizes = padResult(numPoints, result);

        for (uint32_t i=0; i < sizes.size(); ++i) {
            std::cout << "size: " << i << ", count: " << sizes[i] << '\n';
        }
        std::cout << "total grouping/process time (ms): " << groupProcessTime << '\n';
    }




};
#endif //SIGMOD23ANN_SOLUTIONKMEANS_HPP
