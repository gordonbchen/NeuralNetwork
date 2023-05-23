import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    Z -= np.max(Z, axis=0)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def softmax_deriv(Z):
    Z -= np.max(Z, axis=0)

    e_Z = np.exp(Z)
    col_sums = np.sum(e_Z, axis=0)[np.newaxis, :]
    
    return ((col_sums - e_Z) * e_Z) / (col_sums ** 2)


def mean_squared_error(A, Y):
    return np.mean((A - Y) ** 2)

def mean_squared_error_deriv(A, Y):
    return (2 / Y.shape[1]) * (A - Y)


def one_hot_encode(Y):
    encoded = np.zeros((10, Y.size))
    encoded[Y, np.arange(Y.size)] = 1
    return encoded

def one_hot_decode(Z):
    decoded = np.argmax(Z, axis=0)
    return decoded


def get_percent_accuracy(true_output, guess_output):
    return np.mean(true_output == guess_output)
