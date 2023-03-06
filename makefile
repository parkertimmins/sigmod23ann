# CC = clang++
# OPTION = -I./ -DIN_PARALLEL -Xpreprocessor -fopenmp -lomp
# LFLAGS = -std=c++11 -O3 $(OPTION)

CC = g++
OPTION = -I./ -DIN_PARALLEL #-fopenmp
LFLAGS = -std=c++11 -O3 $(OPTION)

all: knng

knng : knn-construction.cc
	$(CC) $(LFLAGS) knn-construction.cc -o $@

clean :
	rm knng
