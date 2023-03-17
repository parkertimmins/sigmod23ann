cmake_minimum_required(VERSION 3.22)
project(sigmod23ann)

set(CMAKE_CXX_STANDARD 20)

#add_compile_options(-O2 -maxv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")


add_executable(sigmod23ann main.cpp)

add_executable(grader grade.cpp)


find_package(TBB REQUIRED)
target_link_libraries(sigmod23ann TBB::tbb)






