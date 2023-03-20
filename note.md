
https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods

All nearest neighbors: 
    https://link.springer.com/article/10.1007/BF02187718
    https://www.cs.umd.edu/gvil/papers/jagan_CG07.pdf


For every node:
    float, uint32_t => 8bytes
    priority queue with (float, uint32_t) x 100 => 8byte x 100 = 800
    set with (uint32_t) x 100 => 400


    ==> 1208 per points
    1,208,000,000
    There must be lots more to fill 8gb




run:
    group size < 1000
    avg percent: 1.16654

run:
    group size < 200
    1 iteration
    percent: 0.282736 

run:
    group size < 200
    2 iteration
    percent: 0.565548

run:
    group size < 200
    10 iteration
    percent: 2.83416 
    time 4m20s


run:
    group size < 200
    7 iteration
    time 60s
    percent: 2.26304


Splits sizes based on power of 2 splits:
0 1000000
1 500000
2 250000
3 125000
4 62500
5 31250
6 15625
7 7812
8 3906
9 1953
10 976
11 488
12 244
13 122
14 61



