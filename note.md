
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
 


===================================================================================================================
Slow Kmean processing speed:
Version: 106e79a (not kmeans)
    Iteration: 0
    group hash time: 369
    histogram split time: 195
    processing time: 8330
    count: 1000000
    --------------------------------------------------------------------------------------------------------
    Iteration: 1
    group hash time: 196
    histogram split time: 88
    processing time: 4225
    count: 1000000
    --------------------------------------------------------------------------------------------------------
    Iteration: 2
    group hash time: 186
    histogram split time: 97
    processing time: 3482
    count: 1000000
    --------------------------------------------------------------------------------------------------------
    Iteration: 3
    group hash time: 196
    histogram split time: 83
    processing time: 3082
    count: 1000000
    --------------------------------------------------------------------------------------------------------
    Iteration: 4
    group hash time: 196
    histogram split time: 76
    processing time: 3186
    count: 1000000

Version: Kmeans

    Iteration: 0
    processing time: 5859
    pair count: 142127330
    count: 1000000
    --------------------------------------------------------------------------------------------------------
    group knn time: 9001
    Iteration: 1
    processing time: 13179
    pair count: 284394198
    count: 2000000
    --------------------------------------------------------------------------------------------------------
    group knn time: 5538
    Iteration: 2
    processing time: 9991
    pair count: 426347408
    count: 3000000
    --------------------------------------------------------------------------------------------------------
    group knn time: 5601
    Iteration: 3
    processing time: 9637
    pair count: 567824012
    count: 4000000
    --------------------------------------------------------------------------------------------------------
    group knn time: 5799
    Iteration: 4
    processing time: 9722
    pair count: 710469898
    count: 5000000

Very silly bug! Processing all ranges multiple times!
