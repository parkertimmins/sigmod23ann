
/**
 *  Naive baseline for construction a KNN Graph.
 *  Randomly select 100 neighbors from a 10k subset.
 */

#include <sys/time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "io.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

#define _INT_MAX 2147483640

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs) {
  float ans = 0.0;
  unsigned lensDim = 100;
  for (unsigned i = 0; i < lensDim; ++i) {
    ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
  }
  return ans;
}

vector<uint32_t> CalculateOneKnn(const vector<vector<float>> &data,
                                 const vector<uint32_t> &sample_indexes,
                                 const uint32_t id) {
  std::priority_queue<std::pair<float, uint32_t>> top_candidates;
  float lower_bound = _INT_MAX;
  for (unsigned i = 0; i < sample_indexes.size(); i++) {
    uint32_t sample_id = sample_indexes[i];
    if (id == sample_id) continue;  // skip itself.
    float dist = EuclideanDistance(data[id], data[sample_id]);

    // only keep the top 100
    if (top_candidates.size() < 100 || dist < lower_bound) {
      top_candidates.push(std::make_pair(dist, i));
      if (top_candidates.size() > 100) {
        top_candidates.pop();
      }

      lower_bound = top_candidates.top().first;
    }
  }

  vector<uint32_t> knn;
  while (!top_candidates.empty()) {
    knn.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(knn.begin(), knn.end());
  return knn;
}

void ConstructKnng(const vector<vector<float>> &data,
                   const vector<uint32_t> &sample_indexes,
                   vector<vector<uint32_t>> &knng) {
  knng.resize(data.size());
#pragma omp parallel for
  for (uint32_t n = 0; n < knng.size(); n++) {
    knng[n] = CalculateOneKnn(data, sample_indexes, n);
  }
  cout << "Construction Done!" << endl;
}

int main(int argc, char **argv) {
  string source_path = "dummy-data.bin";

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  // Read data points
  vector<vector<float>> nodes;
  ReadBin(source_path, nodes);

  // Sample 10,000 points for greedy search
  std::default_random_engine rd;
  std::mt19937 gen(rd());  // Mersenne twister MT19937
  vector<uint32_t> sample_indexes(nodes.size());
  iota(sample_indexes.begin(), sample_indexes.end(), 0);
  shuffle(sample_indexes.begin(), sample_indexes.end(), gen);
  sample_indexes.resize(10000);

  // Knng constuction
  vector<vector<uint32_t>> knng;
  ConstructKnng(nodes, sample_indexes, knng);

  // Save to ouput.bin
  SaveKNNG(knng);

  return 0;
}
