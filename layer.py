import numpy as np

class Layer:
    def __init__(self, n_inputs, n_outputs, activation_function):
        self.activation_function = activation_function
        
        # Initialize empty weights and biases.
        self.weights = np.random.random(n_outputs * n_inputs).reshape(n_outputs, n_inputs)
        self.biases = np.random.random(n_outputs).reshape(n_outputs, 1)
        
        # Initalize empty weight and bias gradient arrays.
        self.weight_gradients = np.zeros((n_outputs, n_inputs))
        self.bias_gradients = np.zeros((n_outputs, 1))
        
    def get_activations(self, inputs):
        """Calculate the layer's activations."""
        activations = self.weights.dot(inputs) + self.biases
        return self.activation_function(activations)
    
    def apply_gradients(self, learning_rate):
        """Apply all gradients."""
        self.weights -= self.weight_gradients * learning_rate
        self.biases -= self.bias_gradients * learning_rate
        