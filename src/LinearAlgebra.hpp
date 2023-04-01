#ifndef SIGMOD23ANN_LINEARALGEBRA_HPP
#define SIGMOD23ANN_LINEARALGEBRA_HPP
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
#include "Constants.hpp"

using std::vector;




float distancePartial(vector<uint32_t>& dimensions, const float* lhs, const float* rhs) {
    float sumDiffs = 0;
    for (auto& i : dimensions) {
        float diff = lhs[i] - rhs[i];
        sumDiffs += diff * diff;
    }
    return sumDiffs;
}

float dotPartial(vector<uint32_t>& dimensions, const float* lhs, const float* rhs) {
    float sum = 0;
    for (auto& i : dimensions) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}





float distance(const float* lhs, const float* rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        __m256 diff = _mm256_sub_ps(ls, rs);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        r += 8;
        l += 8;
    }

    alignas(sizeof(__m256)) float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < dims; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}

double norm(const Vec& vec) {
    float sumSquares = 0.0;
    for (auto& v : vec) {
        sumSquares += (v * v);
    }
    return sqrt(sumSquares);
}

void normalizeInPlace(Vec& vec) {
    double vecNorm = norm(vec);
    for (float& v : vec) {
        v = v / vecNorm;
    }
}

Vec normalize(const Vec& vec) {
    Vec res;
    double vecNorm = norm(vec);
    for (float v : vec) {
        res.push_back(v / vecNorm);
    }
    return vec;
}

Vec randUniformUnitVec(size_t dim=dims) {
    std::mt19937 gen(rd());
    std::normal_distribution<> normalDist(0, 1);

    Vec randVec;
    randVec.reserve(dim);
    while (randVec.size() < dim) {
        auto elementValue = normalDist(gen);
        randVec.emplace_back(elementValue);
    }

    // make unit vector just for good measure
    normalizeInPlace(randVec);
    return randVec;
}


Vec add(const Vec& lhs, const Vec& rhs) {
    auto dim = lhs.size();
    Vec result(dim);
    for (uint32_t i = 0; i < dim; i++) {
        result[i]  = lhs[i] + rhs[i];
    }
    return result;
}

Vec sub(const float* lhs, const float* rhs) {
    uint32_t dim = 100;
    Vec result(dim);
    for (uint32_t i = 0; i < dim; i++) {
        result[i]  = lhs[i] - rhs[i];
    }
    return result;
}

Vec sub(const Vec& lhs, const Vec& rhs) {
    auto dim = lhs.size();
    Vec result(dim);
    for (uint32_t i = 0; i < dim; i++) {
        result[i]  = lhs[i] - rhs[i];
    }
    return result;
}

Vec scalarMult(float c, const Vec& vec) {
    Vec res;
    for (float v : vec) {
        res.push_back(c * v);
    }
    return res;
}

float dot(const float* lhs, const float* rhs) {
    __m256 sum  = _mm256_set1_ps(0);
    auto* r = rhs;
    auto* l = lhs;
    for (uint32_t i = 0; i < 96; i+=8) {
        __m256 rs = _mm256_load_ps(r);
        __m256 ls = _mm256_load_ps(l);
        sum = _mm256_fmadd_ps(rs, ls, sum);
        l += 8;
        r += 8;
    }
    alignas(sizeof(__m256)) float sums[8] = {};
    _mm256_store_ps(sums, sum);
    float ans = 0.0f;
    for (float s: sums) {
        ans += s;
    }
    for (unsigned i = 96; i < dims; ++i) {
        ans += (lhs[i] * rhs[i]);
    }
    return ans;
}


// project v onto u
Vec project(const Vec& u, const Vec& v) {
    return scalarMult(dot(u.data(), v.data()), normalize(u));
}


// assume that argument unit vectors are linearly independent
// also unit vectors
vector<Vec> gramSchmidt(vector<Vec>& v) {
    vector<Vec> u(v.size());
    u[0] = normalize(v[0]);
    for (uint32_t i = 1; i < v.size(); ++i) {
        u[i] = v[i];
        for (uint32_t j = 0; j < i; ++j) {
            u[i] = sub(u[i], project(u[j], v[i]));
        }
        u[i] = normalize(u[i]);
    }
    return u;
}

#endif //SIGMOD23ANN_LINEARALGEBRA_HPP
