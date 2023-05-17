import numpy as np
import sklearn.datasets

from functions import *

from layer import Layer
from neural_network import NeuralNetwork

def load_data():
    data = sklearn.datasets.load_digits()
    X = data["data"] / 16
    y = data["target"]

    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)

    train_inds = inds[:1500]
    test_inds = inds[1500:]

    X_train, y_train = X[train_inds], y[train_inds]
    X_test, y_test = X[test_inds], y[test_inds]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

layers = [
    Layer(64, 12, relu),
    Layer(12, 10, softmax)
]

neural_network = NeuralNetwork(
    layers, sum_squared_error, get_percent_accuracy, 64, 0.1
)
neural_network.train(X_train.T, one_hot_encode(y_train), 10_000)
