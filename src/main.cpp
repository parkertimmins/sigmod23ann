#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include "io.h"
#include "ann.hpp"
//#include "SolutionKmeans.hpp"
#include "SolutionRandomKD.hpp"


/**
 * Docs used
 * https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
 * https://en.wikipedia.org/wiki/Locality-sensitive_hashing
 * http://infolab.stanford.edu/~ullman/mining/2009/similarity3.pdf
 * http://infolab.stanford.edu/~ullman/mining/pdf/cs345-lsh.pdf
 * http://infolab.stanford.edu/~ullman/mining/2008/slides/cs345-lsh.pdf
 * https://users.cs.utah.edu/~jeffp/teaching/cs5955/L6-LSH.pdf
 * http://infolab.stanford.edu/~ullman/mmds/ch3.pdf
 * http://web.mit.edu/andoni/www/papers/cSquared.pdf
 * https://courses.engr.illinois.edu/cs498abd/fa2020/slides/14-lec.pdf
 * https://www.youtube.com/watch?v=yIkyeackISs&ab_channel=SimonsInstitute
 * https://arxiv.org/abs/1501.01062
 * http://www.slaney.org/malcolm/yahoo/Slaney2012(OptimalLSH).pdf
 * https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf
 * https://people.csail.mit.edu/indyk/icm18.pdf
 * https://arxiv.org/pdf/1806.09823.pdf - Approximate Nearest Neighbor Search in High Dimensions
 * https://people.csail.mit.edu/indyk/p117-andoni.pdf
 * https://www.youtube.com/watch?v=cn15P8vgB1A&ab_channel=RioICM2018
 * Nearest-Neighbor Methods in Learning and Vision: Theory and Practice
 * https://hal.inria.fr/inria-00567191/en/
 */



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
  auto startRead = hclock::now();
  auto [points, numPoints] = ReadBinArray(source_path);
#ifdef PRINT_OUTPUT
  std::cout << "read time: " << duration_cast<milliseconds>(hclock::now() - startRead).count() << '\n';
#endif

  // Knng constuction
  vector<vector<uint32_t>> knng(numPoints);
  SolutionRandomKD::constructResult(points, numPoints, knng);

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

