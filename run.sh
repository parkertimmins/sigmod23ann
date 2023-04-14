#!/bin/bash

#make clean
#make
#./knng


mkdir -p build
cd build
time cmake ..
make clean
time make -j$(nproc)
cd ..
./build/sigmod23ann
