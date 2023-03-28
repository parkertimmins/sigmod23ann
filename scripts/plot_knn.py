import numpy as np
import matplotlib.pyplot as plt

tests = [
    (0.072, 365, 275),      # 1knn, 10total iter, group size 200
    (0.064, 369, 288),
    (0.085, 473, 389),      # 1knn, 10total iter, group size 500
    (0.049, 253, 169),       # 3
    (0.024,  216, 133),      # 1
    (0.007	, 201, 117),     # 0
    (0.06, 267.370570, 171), # 5
    (0.073, 395.257150, 308) # 20
]
#plt.scatter(x, y)
#plt.show()

for perc, time, measured_time in tests:
    p_s = perc / time
    p_s2 = perc / measured_time

    print(perc, 1800 * p_s, 1800 * p_s2)
