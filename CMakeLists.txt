cmake_minimum_required(VERSION 3.22)
project(sigmod23ann)

set(CMAKE_CXX_STANDARD 20)


include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

find_package(TBB REQUIRED)

if (DEFINED ENV{USE_AVX512})
    message(STATUS "building make with avx512")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_AVX512=1 -Ofast -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")
else()
    message(STATUS "building make without avx512")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")
endif()



add_subdirectory(thirdparty/robin-map)
#add_subdirectory(thirdparty/hopscotch-map)
#add_subdirectory(thirdparty/sparse-map)

file(GLOB srcFiles "src/*.hpp" "src/*.cpp")
add_library(lib_ann ${srcFiles})
target_link_libraries(lib_ann tsl::robin_map)

add_executable(sigmod23ann src/main.cpp)
target_link_libraries(sigmod23ann lib_ann TBB::tbb tsl::robin_map)

file(GLOB testFiles "test/*.hpp" "test/*.cpp")
add_executable(tests test/TestMain.cpp ${testFiles})
target_link_libraries(tests gtest gtest_main lib_ann TBB::tbb)

file(GLOB srcFiles "src/*.hpp" "src/*.cpp")
add_executable(grader src/grade.cpp src/Constants.hpp)
target_link_libraries(grader lib_ann TBB::tbb)





