from math import exp

# https://hur.st/bloomfilter/?n=300&p=&m=384&k=1

# n: num elements
# k: num hash functions
# p: prob of false positive
# m: bits in filters
def bloom_p(n, m, k): 
    return pow(1 - exp(-k / (m / n)), k)


total_fp = 0
true_neg = 0
table_size = 256
for i in range(100, 200):
    fpr = bloom_p(n=i, m=table_size, k=1)
    total_fp += fpr
    true_neg += (1 - fpr)
    print(i, total_fp, true_neg)


print(total_fp)
