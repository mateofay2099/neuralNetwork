import numpy as np
from trainingSet import X, Y
from neuralNetwork import use, initialize_parameters
from lossFunctions import mse
from visualisation import visualiseData, visualiseErrorProgression

LAYER_DIMENSIONS = [2, 8, 8, 1]
params = initialize_parameters(LAYER_DIMENSIONS)

errors = []

# Training the neural network
for i in range(90000):
    output = use(params, X, Y, 0.00001, train=True)
    if i % 25 == 0:
        error = mse(Y, output)
        errors.append(error)
        print(error)
# visualiseErrorProgression(errors)

# Testing the neural network
data_test = (np.random.rand(1000, 2) * 2) - 1
y = use(params, data_test)

visualiseData(data_test, y)
