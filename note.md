
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
 


