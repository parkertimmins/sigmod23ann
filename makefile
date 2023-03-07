# CC = clang++
# OPTION = -I./ -DIN_PARALLEL -Xpreprocessor -fopenmp -lomp
# LFLAGS = -std=c++11 -O3 $(OPTION)

CC = g++
CXX_FLAGS = -std=c++20 -O3 -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread
#OPTION = -I./ -DIN_PARALLEL #-fopenmp
#CXX_FLAGS = -std=c++20 -O3 $(OPTION) -fPIC -Wall -Wextra -fno-omit-frame-pointer -march=native -Wno-unknown-pragmas -pthread -g

all: knng

knng :main.cpp
	$(CC) $(CXX_FLAGS)  main.cpp -o $@

clean :
	rm knng
