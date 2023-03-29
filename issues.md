

- [x] pca1 based vector splitting
- [x] faster set implementation
    - just use array from heap


- [x] dont use set during group processing, only at end of group merge sets
- [x] dont use set contains during first run
- [ ] try actually using l1 instead of l2
- [ ] try filtering by l1, l2 or l_inf of a small number of columns 
- [ ] optimize splitting, but timing processing and grouping, then picking size based on best
- [ ] threadize the initial build groups step
- [ ] fix timing to be closer to bound by measuring read time and including
- [ ] measure full time for actuall program, and add build estimate by substracting value from listed on page
- [ ] use random subspace, perhaps of dim 20 to do partitioning, random offset multiple of 4 so works with distance 128 easily
- [ ] can group and process be in single thread? problem is updating priority queues become threaded
- [x] try doubling again! also make sure to check low values as  

- [ ] tbb parallelization for processing loop
- [ ] inspect parallelization in 32cpu machine
 



