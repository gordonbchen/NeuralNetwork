import numpy as np
import pandas as pd

from neural_network import NeuralNetwork

def load_data():
    data = pd.read_csv("data/train.csv").values
    
    np.random.shuffle(data)

    X = data[:, 1:]
    y = data[:, 0]

    X_train = X[: 1_000, :]
    y_train = y[: 1_000]

    X_test = X[1_000 :, :]
    y_test = y[1_000 :]

    return X_train, y_train, X_test, y_test

# Load data.
X_train, y_train, X_test, y_test = load_data()

# Create neural network.
nn = NeuralNetwork()
nn.train(X_train, y_train, 10_000)
