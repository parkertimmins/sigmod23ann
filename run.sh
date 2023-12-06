#!/bin/bash



mkdir -p build
cd build
time cmake ..
make clean
time make -j$(nproc)
cd ..
./build/sigmod23ann
