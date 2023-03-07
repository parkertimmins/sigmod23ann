#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>
#include <cstdint>
#include "io.h"


using std::cout;
using std::endl;
using std::string;
using std::vector;

using Vec = vector<float>;

#define _INT_MAX 2147483640

/**
 * Docs used
 * https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
 * https://en.wikipedia.org/wiki/Locality-sensitive_hashing
 */

float distance(const Vec &lhs, const Vec &rhs) {
    float ans = 0.0;
    unsigned lensDim = 100;
    for (unsigned i = 0; i < lensDim; ++i) {
        auto d = (lhs[i] - rhs[i]);
        ans += (d * d);
    }
    return ans;
}

std::default_random_engine rd(123);
vector<float> makeRandoms(size_t len=100) {
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    Vec randVec;
    randVec.reserve(len);
    while (randVec.size() < len) {
        randVec.emplace_back(dist(gen));
    }
    return randVec;
}

float dot(const Vec& lhs, const Vec& rhs) {
    float ans = 0.0;
    for (unsigned i = 0; i < 100; ++i) {
        ans += (lhs[i] * rhs[i]);
    }
    return ans;
}

uint64_t projectRandoms(const vector<Vec>& randVecs, const Vec& values) {
    uint64_t projection = 0;
    for (auto& rv : randVecs) {
        auto dotProduct = dot(values, rv);
        projection <<= 1;
        projection |= (dotProduct > 0);
    }
    return projection; 
}


int numGroupBits = 10;
void constructResult(const vector<Vec>& data, vector<vector<uint32_t>>& result) {
 
    vector<Vec> randVecs;
    while (randVecs.size() < numGroupBits) {
        auto randv = makeRandoms();
        randVecs.push_back(randv);
    }
    
    std::unordered_map<uint64_t, vector<uint32_t>> groups;

    // load vectors in map
    for (uint32_t i = 0, len=data.size(); i < len; ++i) {
        if (i % 100'000 == 0) std::cout << "inserted: " << i << std::endl;

        uint64_t signature = projectRandoms(randVecs, data[i]);
        if (groups.find(signature) == groups.end()) {
            groups[signature] = { i };
        } else {
            groups[signature].push_back(i);
        }
    }
    std::cout << groups.size() << std::endl;

    result.resize(data.size());


    for (uint32_t i = 0, len=data.size(); i < len; ++i) {
        const auto& v = data[i];
        uint64_t signature = projectRandoms(randVecs, v);

        auto& group = groups[signature];
        std::cout << group.size() << std::endl;
    }

}


int main(int argc, char **argv) {
  string source_path = "../dummy-data.bin";

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  // Read data points
  vector<vector<float>> nodes;
  ReadBin(source_path, nodes);


  // Knng constuction
  vector<vector<uint32_t>> knng;
  constructResult(nodes, knng);

  // Save to ouput.bin
  SaveKNNG(knng);

  return 0;
}

