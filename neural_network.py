import numpy as np

import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns

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

        # Save training and testing accuracies across epochs.
        self.training_accuracies = []
        self.testing_accuracies = []
        self.accuracy_resolution = 10

        # Hyperparams.
        self.learning_rate = 0.25
        self.mini_batch_size = 256


    def train(self, X_train, y_train, X_test, y_test, epochs):
        """Train the network for the given epochs."""
        self.training_accuracies = []
        self.testing_accuracies = []

        for epoch in range(epochs):
            for start_index in range(0, y_train.size, self.mini_batch_size):
                mini_batch_X, mini_batch_y = self.get_mini_batch(X_train, y_train, start_index)
                self.gradient_descent(mini_batch_X, mini_batch_y)

            if (epoch % self.accuracy_resolution == 0):
                # Save epoch train and test accuracies.
                training_accuracy = self.get_accuracy(X_train, y_train)
                self.training_accuracies.append(training_accuracy)

                testing_accuracy = self.get_accuracy(X_test, y_test)
                self.testing_accuracies.append(testing_accuracy)

                print(f"Epoch {epoch}: \t\t training accuracy={training_accuracy}")

            # Shuffle training data.
            inds = np.arange(y_train.size)
            np.random.shuffle(inds)
            X_train = X_train[inds, :]
            y_train = y_train[inds]

    
    def get_mini_batch(self, X, y, start_index):
        """Get the mini batch starting with the given index."""
        end_index = start_index + self.mini_batch_size

        mini_batch_X = X[start_index : end_index, :]
        mini_batch_y = y[start_index : end_index]
        return mini_batch_X, mini_batch_y


    def gradient_descent(self, X, y):
        """Gradient descent learning step."""
        Z1, A1, Z2, A2 = self.forward_prop(X)
        dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, y)
        self.apply_gradients(dW1, db1, dW2, db2, self.learning_rate)

    def forward_prop(self, X):
        """Forward-prop to get the network's output."""
        Z1 = np.dot(self.W1, X.T) + self.b1[:, np.newaxis]
        A1 = self.f_act1(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2[:, np.newaxis]
        A2 = self.f_act2(Z2)
        return Z1, A1, Z2, A2
    
    def back_prop(self, Z1, A1, Z2, A2, X, y):
        """Back-prop to find the network's gradient."""
        one_hot_Y = functions.one_hot_encode(y)

        dA2 = self.f_cost_deriv(A2, one_hot_Y)
        dZ2 = dA2 * self.f_act2_deriv(Z2)

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


    def get_predictions(self, X):
        """Get the network's predictions."""
        Z1, A1, Z2, A2 = self.forward_prop(X)
        predictions = functions.one_hot_decode(A2)
        return predictions

    def get_accuracy(self, X, y):
        """Get the network's prediction accuracy."""
        predictions = self.get_predictions(X)
        accuracy = self.f_accuracy(y, predictions)
        return accuracy


    def display_image_predictions(self, X, y):
        """Display the network's predictions."""
        predictions = self.get_predictions(X)

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

    def display_prediction_confusion_matrix(self, X, y):
        """Show network confusion matrix."""
        nums = np.arange(10)

        predictions = self.get_predictions(X)
        confusion_matrix = sklearn.metrics.confusion_matrix(y, predictions, labels=nums)

        sns.heatmap(
            confusion_matrix,
            square=True, annot=True, cbar=False, fmt="d",
            xticklabels=nums, yticklabels=nums,
        )
        plt.xlabel("prediction")
        plt.ylabel("true")

        plt.show()

    def display_accuracies_over_epochs(self):
        """Show training and testing accuracies over epochs."""
        epochs = np.arange(0, len(self.training_accuracies) * self.accuracy_resolution, self.accuracy_resolution)
        plt.plot(
            epochs,
            self.training_accuracies,
            label="Training accuracy",
            color="blue"
        )

        plt.plot(
            epochs,
            self.testing_accuracies,
            label="Testing accuracy",
            color="red"
        )

        plt.title("% Training and Testing accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("% Accuracy")

        plt.legend(loc="lower right")

        plt.show()
