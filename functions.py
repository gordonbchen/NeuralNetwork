import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    safe_Z = Z - np.max(Z, axis=0)
    e_Z = np.exp(safe_Z)
    return e_Z / np.sum(e_Z, axis=0)

def softmax_deriv(Z):
    safe_Z = Z - np.max(Z, axis=0)

    e_Z = np.exp(safe_Z)
    col_sums = np.sum(e_Z, axis=0)

    a = col_sums - e_Z
    
    return (a * e_Z) / (col_sums ** 2)


def mean_squared_error(A, Y):
    return np.mean((A - Y) ** 2)

def mean_squared_error_deriv(A, Y):
    return (2 / Y.shape[0]) * (A - Y)


def one_hot_encode(Y):
    encoded = np.zeros((10, Y.size))
    encoded[Y, np.arange(Y.size)] = 1
    return encoded

def one_hot_decode(Z):
    decoded = np.argmax(Z, axis=0)
    return decoded


def get_percent_accuracy(true_output, guess_output):
    return np.mean(true_output == guess_output) * 100.0
