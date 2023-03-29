#ifndef SIGMOD23ANN_CONSTANTS_HPP
#define SIGMOD23ANN_CONSTANTS_HPP

#include <cstdint>
#include <limits>
#include <random>
#include <boost/align/aligned_allocator.hpp>
#include <immintrin.h>



using Range = std::pair<uint32_t, uint32_t>;
std::pair<float, float> startBounds = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};


template<class T, std::size_t Alignment = sizeof(__m256)>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, Alignment> >;
using Vec = aligned_vector<float>;

#define PRINT_OUTPUT

std::default_random_engine rd(123);
const uint32_t dims = 100;
const uint32_t k = 100;


#endif //SIGMOD23ANN_CONSTANTS_HPP
