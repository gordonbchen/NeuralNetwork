class NeuralNetwork:
    def __init__(self, layers, cost_function, accuracy_function, batch_size, learning_rate):
        self.layers = layers
        
        self.cost_function = cost_function
        self.accuracy_function = accuracy_function
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def train(self, inputs, true_outputs, iterations):
        """Learn for the specified number of iterations."""
        for i in range(iterations):
            if (i % 100 == 0):
                accuracy = self.get_accuracy(inputs, true_outputs)
                cost = self.get_cost(inputs, true_outputs)
                print(f"{i}. accuracy={accuracy}\tcost={cost}")
            
            input_batch, true_output_batch = self.get_training_batch(i, inputs, true_outputs)
            self.learning_step(input_batch, true_output_batch)
            
    def get_accuracy(self, inputs, true_outputs):
        """Get the network's accuracy."""
        guess_outputs = self.get_guess_outputs(inputs)
        return self.accuracy_function(true_outputs, guess_outputs)
        
    def get_guess_outputs(self, inputs):
        """Get the network's forward-propagation guess output."""
        layer_activations = self.get_layer_activations(inputs)
        return layer_activations[-1]
    
    def get_layer_activations(self, inputs):
        layer_activations = [inputs]
        for layer in self.layers:
            activations = layer.get_activations(layer_activations[-1])
            layer_activations.append(activations)
        return layer_activations[1:]

    def get_training_batch(self, iteration, inputs, true_outputs):
        """Return the next training batch."""
        start = (iteration * self.batch_size) % inputs.shape[1]
        end = min(start + self.batch_size, inputs.shape[1])
        return inputs[:, start : end], true_outputs[:, start : end]
    
    def learning_step(self, inputs, true_outputs):
        """Do a learning step. Calculate and apply gradients."""
        self.calc_gradients(inputs, true_outputs)
        self.apply_gradients()
    
    def calc_gradients(self, inputs, true_outputs):
        """Calculate gradients for all layers."""
        h = 0.000001
        base_cost = self.get_cost(inputs, true_outputs)
        
        for layer in self.layers:
            self.calc_weight_gradients(layer, inputs, true_outputs, base_cost, h)
            self.calc_bias_gradients(layer, inputs, true_outputs, base_cost, h)
            
    def get_cost(self, inputs, true_outputs):
        """Get the network cost."""
        guess_outputs = self.get_guess_outputs(inputs)
        return self.cost_function(true_outputs, guess_outputs)
            
    def calc_weight_gradients(self, layer, inputs, true_outputs, base_cost, h):
        """Calculate weight gradients."""
        for nrow in range(layer.weights.shape[0]):
            for ncol in range(layer.weights.shape[1]):
                base_weight = layer.weights[nrow, ncol]
                layer.weights[nrow, ncol] += h

                new_cost = self.get_cost(inputs, true_outputs)
                layer.weight_gradients[nrow, ncol] = (new_cost - base_cost) / h
                
                layer.weights[nrow, ncol] = base_weight
                
    def calc_bias_gradients(self, layer, inputs, true_outputs, base_cost, h):
        """Calculate bias gradients."""
        for i in range(layer.biases.shape[0]):
            base_bias = layer.biases[i, 0]
            layer.biases[i, 0] += h

            new_cost = self.get_cost(inputs, true_outputs)
            layer.bias_gradients[i, 0] = (new_cost - base_cost) / h

            layer.biases[i, 0] = base_bias
            
    def apply_gradients(self):
        """Apply gradients for each layer."""
        for layer in self.layers:
            layer.apply_gradients(self.learning_rate)
            