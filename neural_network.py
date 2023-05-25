import numpy as np
import matplotlib.pyplot as plt

import functions

class NeuralNetwork:
    def __init__(self):
        """Initialize weights and biases, and hyperparameters."""
        # Layer 1.
        l1_inputs, l1_nodes = 784, 12
        self.W1 = np.random.random(l1_nodes * l1_inputs).reshape(l1_nodes, l1_inputs) - 0.5
        self.b1 = np.random.random(l1_nodes) - 0.5

        self.f_act1 = functions.ReLU
        self.f_act1_deriv = functions.ReLU_deriv

        # Layer 2.
        l2_inputs, l2_nodes = 12, 10
        self.W2 = np.random.random(l2_nodes * l2_inputs).reshape(l2_nodes, l2_inputs) - 0.5
        self.b2 = np.random.random(l2_nodes) - 0.5

        self.f_act2 = functions.softmax
        self.f_act2_deriv = functions.softmax_deriv

        # Cost and accuracy functions.
        self.f_cost = functions.mean_squared_error
        self.f_cost_deriv = functions.mean_squared_error_deriv

        self.f_accuracy = functions.get_percent_accuracy

        # Hyperparams.
        self.learning_rate = 0.1


    def train(self, X, y, epochs):
        """Train the network for the given epochs."""
        for epoch in range(epochs):
            # TODO: mini-batches.
            self.gradient_descent(X, y)

            # Show accuracy and cost.
            if (epoch % 10 == 0):
                cost, accuracy = self.get_cost_accuracy(X, y)
                print(f"Epoch {epoch}: \t\t accuracy={accuracy} \t\t cost={cost}")

            # Shuffle data.
            inds = np.arange(y.size)
            np.random.shuffle(inds)
            X = X[inds, :]
            y = y[inds]


    def gradient_descent(self, X, y):
        """Gradient descent learning step."""
        Z1, A1, Z2, A2 = self.forward_prop(X)
        dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, y)
        self.apply_gradients(dW1, db1, dW2, db2, self.learning_rate)

    def forward_prop(self, X):
        """Forward-prop to get the network's output."""
        X = X.T

        Z1 = np.dot(self.W1, X) + self.b1[:, np.newaxis]
        A1 = self.f_act1(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2[:, np.newaxis]
        A2 = self.f_act2(Z2)
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
        dZ1 = dA1 * self.f_act1_deriv(Z1)

        dW1 = np.dot(dZ1, X) / y.size
        db1 = np.mean(dZ1, axis=1)
        return dW1, db1, dW2, db2
    
    def apply_gradients(self, dW1, db1, dW2, db2, learning_rate):
        """Apply calculated gradients with a learning rate."""
        self.W1 -= dW1 * learning_rate
        self.b1 -= db1 * learning_rate
        self.W2 -= dW2 * learning_rate
        self.b2 -= db2 * learning_rate


    def get_cost_accuracy(self, X, y):
        Z1, A1, Z2, A2 = self.forward_prop(X)

        predictions = functions.one_hot_decode(A2)
        accuracy = self.f_accuracy(y, predictions)

        cost = self.f_cost(A2, y)
        return cost, accuracy


    def display_image_predictions(self, X, y):
        """Display the network's predictions."""
        Z1, A1, Z2, A2 = self.forward_prop(X)
        predictions = functions.one_hot_decode(A2)

        fig, axs = plt.subplots(
            nrows=10, ncols=10,
            figsize=(15, 15),
            subplot_kw=dict(xticks=[], yticks=[]),
            gridspec_kw=dict(hspace=0, wspace=0)
        )
        for (i, axi) in enumerate(axs.flat):
            axi.imshow(X[i, :].reshape(28, 28), cmap="binary")
            axi.text(
                x=0.05, y=0.05,
                s=str(predictions[i]),
                c=("green" if predictions[i] == y[i] else "red"),
                transform=axi.transAxes
            )

        plt.show()
