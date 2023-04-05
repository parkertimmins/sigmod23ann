#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"
#include "Constants.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;


/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNNG(const std::vector<std::vector<uint32_t>> &knng,
              const std::string &path = "output.bin") {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  const int K = 100;
  std::cout << "Saving KNN Graph (" << knng.size() << " X 100) to " << path
            << std::endl;
  assert(knng.front().size() == K);
  for (unsigned i = 0; i < knng.size(); ++i) {
    auto const &knn = knng[i];
    ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
  }
  ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             std::vector<Vec> &data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;

  const int num_dimensions = 100;
  Vec buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    Vec row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }
    data[counter++] = std::move(row);
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

std::pair<float(*)[112], uint32_t> ReadBinArray(const std::string &file_path) {

    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t numPoints;  // num of points

    ifs.read((char *)&numPoints, sizeof(uint32_t));
    std::cout << "# of points: " << numPoints << std::endl;

    float (*points)[112] = static_cast<float(*)[112]>(aligned_alloc(64, numPoints * I112 * sizeof(float)));

    const int num_dimensions = 100;
    Vec buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
        float* row = points[counter];
        for (int d = 0; d < num_dimensions; d++) {
          row[d] = static_cast<float>(buff[d]);
        }
        counter++;
    }

    ifs.close();
    std::cout << "Finish Reading Data" << endl;

    return std::make_pair(points, numPoints);
}