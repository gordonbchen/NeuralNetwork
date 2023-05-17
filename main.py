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

# Load data.
X_train, y_train, X_test, y_test = load_data()

# Create neural network.
layers = [
    Layer(64, 12, relu),
    Layer(12, 10, softmax)
]

neural_network = NeuralNetwork(
    layers, mean_squared_error, get_percent_accuracy, 64, 0.1
)

# Load in previous weights and biases.
file_name = "trained_params.json"
neural_network.load_trained_params(file_name)

# Train.
neural_network.train(X_train.T, one_hot_encode(y_train), 5_000)

# Save trained params.
neural_network.save_trained_params(file_name)

# Show network outputs.
neural_network.show_guesses(X_test.T, one_hot_encode(y_test))
