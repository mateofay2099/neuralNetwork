import numpy as np
from sklearn.datasets import make_gaussian_quantiles

N = 1000
X, Y = make_gaussian_quantiles(mean=None, cov=0.1, n_samples=N,
                               n_features=2, n_classes=2, shuffle=True, random_state=None)

Y = Y[:, np.newaxis]
