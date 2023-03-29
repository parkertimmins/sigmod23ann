#include "gtest/gtest.h"
#include "../src/KnnSets.hpp"
#include <utility>
#include <random>

/////////  KnnSetScannable /////////////////////////////////////////////////////////
TEST(KnnSetScannable, containsNotFull){
    KnnSetScannable ks;
    ks.append(std::make_pair(0.1f, 99));
    ASSERT_TRUE(ks.contains(99));
    ASSERT_FALSE(ks.contains(100));
}

TEST(KnnSetScannable, containsFull){
    KnnSetScannable ks;
    for (uint32_t i = 0; i < 100; ++i) {
        ks.append(std::make_pair(0.01 * i, i));
    }

    ASSERT_TRUE(ks.contains(50));
    ASSERT_FALSE(ks.contains(100));
}

TEST(KnnSetScannable, lowerBoundSetOnFirst100){
    KnnSetScannable ks;

    for (uint32_t i = 0; i < 100; ++i) {
        ks.addCandidate(i, 0.01f * i);
        ASSERT_FLOAT_EQ(ks.lower_bound, 0.01 * i);
    }
}

TEST(KnnSetScannable, lowerBoundSetAfter100){
    KnnSetScannable ks;

    vector<pair<float, uint32_t>> data;
    for (uint32_t i = 0; i < 100; ++i) {
        data.emplace_back(0.01 * i, i);
    }
    std::default_random_engine rand(123);
    std::shuffle(data.begin(), data.end(), rand);

    for (auto& [dist, id] : data) {
        ks.addCandidate(id, dist);
    }

    ks.addCandidate(100, 0.0001); // id not present, and small dist
    ASSERT_FLOAT_EQ(ks.lower_bound, 0.01 * 98); // largest previous was removed;
}

TEST(KnnSetScannable, finalize){
    KnnSetScannable ks;

    // More than 100 items so exercise both paths
    vector<pair<float, uint32_t>> data;
    for (uint32_t i = 0; i < 1000; ++i) {
        data.emplace_back(0.01 * i, i);
    }
    std::default_random_engine rand(123);
    std::shuffle(data.begin(), data.end(), rand);

    for (auto& [dist, id] : data) {
        ks.addCandidate(id, dist);
    }
    
    auto finalResult = ks.finalize();
    for (uint32_t i = 0; i < 100; ++i) {
        ASSERT_EQ(i, finalResult[i]);
    }
}


/////////  KnnSetScannableSimd /////////////////////////////////////////////////////////
TEST(KnnSetScannableSimd, containsNotFull){
    KnnSetScannableSimd ks;
    ks.append(99, 0.01f);
    ASSERT_TRUE(ks.contains(99));
    ASSERT_FALSE(ks.contains(100));
}

TEST(KnnSetScannableSimd, containsFull){
    KnnSetScannableSimd ks;

    vector<pair<float, uint32_t>> data;
    for (uint32_t i = 0; i < 100; ++i) {
        data.emplace_back(0.01 * i, i);
    }
    std::default_random_engine rand(123);
    std::shuffle(data.begin(), data.end(), rand);
      for (auto& [dist, id] : data) {
        ks.addCandidate(id, dist);
    }

    ASSERT_TRUE(ks.containsFull(50));
    ASSERT_FALSE(ks.containsFull(100));
}

TEST(KnnSetScannableSimd, lowerBoundSetOnFirst100){
    KnnSetScannableSimd ks;

    for (uint32_t i = 0; i < 100; ++i) {
        ks.addCandidate(i, 0.01f * i);
        ASSERT_FLOAT_EQ(ks.lower_bound, 0.01 * i);
    }
}

TEST(KnnSetScannableSimd, lowerBoundSetAfter100){
    KnnSetScannableSimd ks;

    vector<pair<float, uint32_t>> data;
    for (uint32_t i = 0; i < 100; ++i) {
        data.emplace_back(0.01 * i, i);
    }
    std::default_random_engine rand(123);
    std::shuffle(data.begin(), data.end(), rand);

    for (auto& [dist, id] : data) {
        ks.addCandidate(id, dist);
    }

    ks.addCandidate(100, 0.0001); // id not present, and small dist
    ASSERT_FLOAT_EQ(ks.lower_bound, 0.01 * 98); // largest previous was removed;
}

TEST(KnnSetScannableSimd, finalize){
    KnnSetScannableSimd ks;

    // More than 100 items so exercise both paths
    vector<pair<float, uint32_t>> data;
    for (uint32_t i = 0; i < 1000; ++i) {
        data.emplace_back(0.01 * i, i);
    }
    std::default_random_engine rand(123);
    std::shuffle(data.begin(), data.end(), rand);

    for (auto& [dist, id] : data) {
        ks.addCandidate(id, dist);
    }

    auto finalResult = ks.finalize();
    for (uint32_t i = 0; i < 100; ++i) {
        ASSERT_EQ(i, finalResult[i]);
    }
}