import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x -= np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def mean_squared_error(true_output, guess_output):
    return np.mean((true_output - guess_output) ** 2)

def sum_squared_error(true_output, guess_output):
    return np.sum((true_output - guess_output) ** 2)

def one_hot_encode(x):
    encoded = np.zeros((10, x.size))
    encoded[x, np.arange(x.size)] = 1
    return encoded

def one_hot_decode(x):
    decoded = np.argmax(x, axis=0)
    return decoded

def get_percent_accuracy(true_output, guess_output):
    return np.mean(true_output == guess_output)
