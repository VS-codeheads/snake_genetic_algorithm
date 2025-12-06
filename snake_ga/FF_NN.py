# -----------------------------------------
# Feed-Forward Neural Network for Snake AI
# -----------------------------------------

import numpy as np

n_x = 7 # Number of input layers
n_h = 9 # Number of hidden layers 1
n_h2 = 15 # Number of hidden layers 2
n_y = 3 # Number of output layers

W1_shape = (n_h, n_x) # Weights between input layer and hidden layer I.
W2_shape = (n_h2, n_h) # Weights between hidden layer I and hidden layer II.
W3_shape = (n_y, n_h2) # Weights between hidden layer II and output layer.

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)     # numerisk stabilisering
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def decode_weights(individual):
    # safer slicing using cursors
    # beregne indexer med akkumulator-variabel i stedet for at gentage produktet flere steder
    idx = 0
    W1_size = W1_shape[0] * W1_shape[1]
    W1 = individual[0:W1_shape[0] * W1_shape[1]]
    
    W2 = individual[W1_shape[0] * W1_shape[1]:W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]]
    W3 = individual[W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]:]
    return (W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]), W3.reshape(W3_shape[0], W3_shape[1]))

def forward_propagation(X, individual):
    W1, W2, W3 = decode_weights(individual)
    # Using Tanh activation function, because it is zero-centered and helps with convergence

    Z1 = np.matmul(W1, X.T)
    A1 = tahn(Z1)

    Z2 = np.matmul(W2, A1)
    A2 = tahn(Z2)

    Z3 = np.matmul(W3, A2)
    A3 = softmax(Z3.T)

    return A3