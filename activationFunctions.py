import numpy as np


def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)
