import numpy as np
from activationFunctions import sigmoid, relu
from lossFunctions import mse


def initialize_parameters(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)
    for l in range(0, L - 1):
        parameters['W' + str(l + 1)] = (np.random.randn(layer_dimensions[l],
                                                        layer_dimensions[l + 1]) * 2) - 1
        parameters['b' + str(l + 1)] = (np.random.randn(1,
                                                        layer_dimensions[l + 1]) * 2) - 1
    return parameters


def use(params, x_data, y_data=[], learning_rate=0.0001, train=False):

    # Forward propagation
    params['A0'] = x_data

    params['Z1'] = params['A0']@params['W1'] + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = params['A1']@params['W2'] + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = params['A2']@params['W3'] + params['b3']
    # Sigmoid fn to get binary output for classification
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if train:
        # Backpropagation

        # Delta obtained from derivative of the loss fn and derivative of the activation fn in the last layer
        params['dZ3'] = mse(y_data, output, derivate=True) * \
            sigmoid(params['A3'], derivate=True)
        # Transpose the activation matrix to do the dot product and get the delta of the weights
        params['dW3'] = params['A2'].T @ params['dZ3']

        # For previous layers, we propagate the delta from the next layer
        params['dZ2'] = params['dZ3'] @ params['W3'].T * \
            relu(params['A2'], derivate=True)
        params['dW2'] = params['A1'].T @ params['dZ2']

        params['dZ1'] = params['dZ2'] @ params['W2'].T * \
            relu(params['A1'], derivate=True)
        params['dW1'] = params['A0'].T @ params['dZ1']

        # Update weights and bias with Gradient Descent
        params['W3'] = params['W3'] - params['dW3'] * learning_rate
        params['b3'] = params['b3'] - \
            np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate

        params['W2'] = params['W2'] - params['dW2'] * learning_rate
        params['b2'] = params['b2'] - \
            np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate

        params['W1'] = params['W1'] - params['dW1'] * learning_rate
        params['b1'] = params['b1'] - \
            np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output
