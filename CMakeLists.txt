cmake_minimum_required(VERSION 3.22)
project(sigmod23ann)

set(CMAKE_CXX_STANDARD 20)
set(MKL_DIR "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
set(MKLROOT "/opt/intel/oneapi/mkl/latest")

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

find_package(TBB REQUIRED)
#find_package(MKL CONFIG REQUIRED)
find_package(BLAS REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DMKL_ILP64  -m64  -I${MKLROOT}/include")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -I/opt/intel/oneapi/compiler/2023.0.0/linux/include")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -I/opt/intel/oneapi/compiler/2023.0.0/linux/include/sycl/")
#add_link_options(" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl")


file(GLOB srcFiles "src/*.hpp" "src/*.cpp")

add_library(lib_ann ${srcFiles})


add_executable(sigmod23ann src/main.cpp)
target_link_libraries(sigmod23ann lib_ann TBB::tbb openblas)

#target_link_libraries(sigmod23ann lib_ann TBB::tbb $<LINK_ONLY:MKL::MKL>)
#target_compile_options(sigmod23ann  PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
#target_include_directories(sigmod23ann PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)


file(GLOB testFiles "test/*.hpp" "test/*.cpp")
add_executable(tests test/TestMain.cpp ${testFiles})
target_link_libraries(tests gtest gtest_main lib_ann TBB::tbb)

file(GLOB srcFiles "src/*.hpp" "src/*.cpp")
add_executable(grader src/grade.cpp src/Constants.hpp)
target_link_libraries(grader lib_ann TBB::tbb)





