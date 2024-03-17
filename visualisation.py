import numpy as np
import matplotlib.pyplot as plt


def visualiseErrorProgression(errors):
    plt.plot(errors)
    plt.show()


def visualiseData(X, Y):
    y = np.where(Y > 0.5, 1, 0)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.show()
