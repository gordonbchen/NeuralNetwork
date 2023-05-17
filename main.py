import numpy as np
import sklearn.datasets

import json

from functions import *

from layer import Layer
from neural_network import NeuralNetwork

# Load data.
data = sklearn.datasets.load_digits()
X = data["data"] / 16
y = data["target"]

inds = np.arange(X.shape[0])
np.random.shuffle(inds)

train_inds = inds[:1500]
test_inds = inds[1500:]

X_train, y_train = X[train_inds], y[train_inds]
X_test, y_test = X[test_inds], y[test_inds]

# Create neural netowrk and train.
layers = [
    Layer(64, 12, relu),
    Layer(12, 10, softmax)
]

neural_network = NeuralNetwork(
    layers, mean_squared_error, get_percent_accuracy, 64, 0.1
)
neural_network.train(X_train.T, one_hot_encode(y_train), 10_000)

# Save trained parameters.
trained_params = {
    "layer_1" : {
        "weights" : neural_network.layers[0].weights.tolist(),
        "biases" : neural_network.layers[0].biases.tolist()
    },
    "layer_2" : {
        "weights" : neural_network.layers[1].weights.tolist(),
        "biases" : neural_network.layers[1].biases.tolist()
    }
}

file_name = "trained_params.json"
with open(file_name, mode='w') as f:
    json.dump(trained_params, f)
