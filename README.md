
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
[https://inria.hal.science/inria-00567191/document] provides a comparison of several locality sensitive hashing techniques, a few of which I tried. 
The best performing methods was K-Means based splitting.
The method splits the space into a predefined constant number of buckets using the K-means algorithm.
The process is then repeated recursively until a given bucket size is reached.
Surprisingly, the best performing K value was 2, and the best number of iterations per K-means run was 2.
Using a K value of 2 allows a useful optimization.
The K-means algorithm requires testing the distance of every point against each of the K cluster centers.
When only 2 clusters are used, the two distance evaluations can be replaced with a single dot-product.
This is because the plane which is equidistant from both points may be precomputed, an a single dot-product is required to determine on which side of the plane a given points lies.


The code to perform this 2-cluster K-means splitting can be seen at https://github.com/parkertimmins/sigmod23ann/blob/main/src/SolutionKmeans.hpp#L440 . 






    
    


