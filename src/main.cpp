#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include "io.h"
#include "ann.hpp"


using hclock = std::chrono::high_resolution_clock;
using std::vector;

int main(int argc, char **argv) {
  auto startTime = hclock::now();

  string source_path = "dummy-data.bin";

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  // Read data points
  vector<Vec> nodes;

  auto startRead = hclock::now();
  ReadBin(source_path, nodes);
#ifdef PRINT_OUTPUT
  std::cout << "read time: " << duration_cast<milliseconds>(hclock::now() - startRead).count() << '\n';
#endif

  // Knng constuction
  vector<vector<uint32_t>> knng(nodes.size());
  constructResultSplitting(nodes, knng);

  // Save to ouput.bin
  auto startSave = hclock::now();
  SaveKNNG(knng);

#ifdef PRINT_OUTPUT
  std::cout << "save time: " << duration_cast<milliseconds>(hclock::now() - startSave).count() << '\n';
  auto totalDuration = duration_cast<milliseconds>(hclock::now() - startTime).count();
  std::cout << "total time (ms): " << totalDuration << '\n';
#endif
  return 0;
}

