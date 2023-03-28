import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

vecs = np.fromfile("contest-data-release-1m.bin", dtype=np.dtype('<f4'), offset=4)
#print(vecs[0:10])
vecs = np.reshape(vecs, (1000000, 100))




plt.hist(vecs[1, :], bins='auto')
plt.show()


def run_pca():
    pca = PCA()
    pca.fit(vecs)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

run_pca()

