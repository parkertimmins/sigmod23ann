
This is a solution to the [Sigmod 2023 programming contest](http://sigmod2023contest.eastus.cloudapp.azure.com/index.shtml)

It achieved 8th place with a recall of over 94%.

Problem Description:
    Given 10 million points in 100-dimension real space. 
    For each point, approximately find the nearest 100 points, using Euclidean distance.
    Time limit is 30 minutes. 
    Evaluated on recall, eg percent of the 100*10M correct points that are actually returned. Exact formula in the above (seemingly dead link)


Solution:
My solution uses locality sensitve hashing to repeatedly split space into buckets of likely-nearby points.
Then for each bucket do a pairwise comparison of all points in the bucket.
For each point, store the nearest 100 points found so far.
Repeat this process as many times as time permits.
After repeating this process many times, I perform a final step that checks if neighbors of neighbors are nearby.

Space Splitting
[Locality sensitive hashing: a comparison of hash function types and querying mechanisms](https://inria.hal.science/inria-00567191/document) provides a comparison of several locality sensitive hashing techniques, a few of which I tried. 
The best performing methods was K-Means based splitting.
The method splits the space into a predefined constant number of buckets using the K-means algorithm.
The process is then repeated recursively until a given bucket size is reached.
Surprisingly, the best performing K value was 2, and the best number of iterations per K-means run was 2.
Using a K value of 2 allows a useful optimization.
The K-means algorithm requires testing the distance of every point against each of the K cluster centers.
When only 2 clusters are used, the two distance evaluations can be replaced with a single dot-product.
This is because the plane which is equidistant from both points may be precomputed, an a single dot-product is required to determine on which side of the plane a given points lies.

Parallelism:
All [linear algebra code](https://github.com/parkertimmins/sigmod23ann/blob/main/src/LinearAlgebra.hpp#L53) was vectorized with SIMD instrinsics.
The use of the recursive K-means splitting complicated parallelization. High in the tree, when there are very few groups, we want to parallelize the function calls internally. 
For example, when splitting into the first two clusters, all threads should be used to compute which cluster a given points belongs to. This can be done very easily by partition the points into sets and processing each set on a separate thread.
When the data has been split in to many clusters, parallelization becomes more complicated. As single function calls may be processing a small number of points it is inefficient to use all threads within a single Kmean split function call. It is better to use separate function for separate split calls. This allows work to not be overly granular, and keeps data thread-local. But allowing multiple threads to process different instances of a recursive function is not simple. Thankfully, the [TBB library](https://github.com/oneapi-src/oneTBB) provides a solution. We can use [parallel_for](https://github.com/parkertimmins/sigmod23ann/blob/main/src/SolutionKmeans.hpp#L227) for perform parallelism within split function calls, and [parallel_invoke](https://github.com/parkertimmins/sigmod23ann/blob/main/src/SolutionKmeans.hpp#L307) to run recursive function instances in parallel. The TBB thread scheduler balances the work between the two method since they are called within the same function.

Navigating Codebase:
As this code was written for a contest, its a bit messy, with lots of commented out and duplicated code from different versions.
The [constructResult](https://github.com/parkertimmins/sigmod23ann/blob/main/src/SolutionKmeans.hpp#L558) function is the main entry point to the code.
Perhaps more interestingly, the code to perform this 2-cluster K-means splitting can be seen in the [splitKmeansBinaryProcess](https://github.com/parkertimmins/sigmod23ann/blob/main/src/SolutionKmeans.hpp#L142) function.






    
    


