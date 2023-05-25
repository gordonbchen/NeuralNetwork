import numpy as np

import functions

class NeuralNetwork:
    def __init__(self):
        """Initialize weights and biases, and hyperparameters."""
        # Layer 1.
        self.W1 = np.random.random(12 * 784).reshape(12, 784) - 0.5
        self.b1 = np.random.random(12) - 0.5

        self.f1 = functions.ReLU
        self.f1_deriv = functions.ReLU_deriv

        # Layer 2.
        self.W2 = np.random.random(10 * 12).reshape(10, 12) - 0.5
        self.b2 = np.random.random(10) - 0.5

        self.f2 = functions.softmax
        self.f2_deriv = functions.softmax_deriv

        # Cost and accuracy functions.
        self.f_cost = functions.mean_squared_error
        self.f_cost_deriv = functions.mean_squared_error_deriv

        self.f_accuracy = functions.get_percent_accuracy

        # Hyperparams.
        self.learning_rate = 0.1

    def forward_prop(self, X):
        """Forward-prop to get the network's output."""
        X = X.T

        Z1 = np.dot(self.W1, X) + self.b1[:, np.newaxis]
        A1 = self.f1(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2[:, np.newaxis]
        A2 = self.f2(Z2)
        return Z1, A1, Z2, A2
    
    def back_prop(self, Z1, A1, Z2, A2, X, y):
        """Back-prop to find the network's gradient."""
        one_hot_Y = functions.one_hot_encode(y)

        # TODO: cost function and softmax derivatives.
        # dA2 = self.f_cost_deriv(A2, one_hot_Y)
        # dZ2 = self.f2_deriv(dA2)

        dZ2 = A2 - one_hot_Y

        dW2 = np.dot(dZ2, A1.T) / y.size
        db2 = np.mean(dZ2, axis=1)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.f1_deriv(Z1)

        dW1 = np.dot(dZ1, X) / y.size
        db1 = np.mean(dZ1, axis=1)
        return dW1, db1, dW2, db2
    
    def apply_gradients(self, dW1, db1, dW2, db2, learning_rate):
        """Apply calculated gradients with a learning rate."""
        self.W1 -= dW1 * learning_rate
        self.b1 -= db1 * learning_rate
        self.W2 -= dW2 * learning_rate
        self.b2 -= db2 * learning_rate
    
    def gradient_descent(self, X, y):
        """Gradient descent learning step."""
        Z1, A1, Z2, A2 = self.forward_prop(X)
        dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, y)
        self.apply_gradients(dW1, db1, dW2, db2, self.learning_rate)

    def train(self, X, y, epochs):
        """Train the network for the given epochs."""
        for epoch in range(epochs):
            # TODO: mini-batches.
            self.gradient_descent(X, y)

            # Show accuracy and cost.
            if (epoch % 10 == 0):
                Z1, A1, Z2, A2 = self.forward_prop(X)

                predictions = functions.one_hot_decode(A2)
                accuracy = self.f_accuracy(y, predictions)

                cost = self.f_cost(A2, y)
                print(f"Epoch {epoch}: \t\t accuracy={accuracy} \t\t cost={cost}")

            # Shuffle data.
            inds = np.arange(y.size)
            np.random.shuffle(inds)
            X = X[inds, :]
            y = y[inds]
