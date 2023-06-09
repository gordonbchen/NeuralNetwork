import numpy as np
import pandas as pd

from neural_network import NeuralNetwork

def load_data():
    data = pd.read_csv("data/train.csv").values
    
    np.random.shuffle(data)

    X = data[:, 1:]
    X = X / X.max()

    y = data[:, 0]

    X_train = X[: 2_000, :]
    y_train = y[: 2_000]

    X_test = X[2_000 :, :]
    y_test = y[2_000 :]

    return X_train, y_train, X_test, y_test

# Load data.
X_train, y_train, X_test, y_test = load_data()

# Create neural network.
nn = NeuralNetwork()

# Train the network.
nn.train(X_train, y_train, X_test, y_test, 500)

# Show final test set accuracy.
test_set_accuracy = nn.get_accuracy(X_test, y_test)
print(f"\nTest set accuracy: {test_set_accuracy}")

# Show predictions.
nn.display_image_predictions(X_test, y_test)
nn.display_prediction_confusion_matrix(X_test, y_test)
nn.display_accuracies_over_epochs()
