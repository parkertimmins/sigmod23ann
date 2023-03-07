cmake_minimum_required(VERSION 3.23)
project(sigmod23ann)

set(CMAKE_CXX_STANDARD 20)

#add_compile_options(-O2 -maxv)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")


add_executable(sigmod23ann main.cpp)
