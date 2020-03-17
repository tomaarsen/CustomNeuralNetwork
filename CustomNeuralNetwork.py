
import utils
import random, time
import numpy as np
from csv import reader
from math import exp

class Node:
    # Class storing a single node
    def __init__(self, n_weights):
        self.weight = [random.random() for _ in range(n_weights)]
        self.output = None
        self.delta = None

class Network:
    # Class storing a network of nodes in self.layers
    # If n_hidden_layers is [5, 3, 5] and n_outputs is 3, then self.layers
    # will look like [[N, N, N, N, N], [N, N, N], [N, N, N, N, N], [N, N, N]]
    # with N as a Node object
    def __init__(self, n_inputs, n_hidden_layers, n_outputs):
        self.layers = list()
        for i_layer, n_hidden in enumerate(n_hidden_layers + [n_outputs]):
            self.layers.append([])
            for i_node in range(n_hidden):
                self.layers[i_layer].append(Node(len(self.layers[i_layer - 1]) if i_layer > 0 else n_inputs))

class NeuralNetwork:

    def __init__(self, n_inputs, n_hidden_layers, n_outputs):
        # Set Variables
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.n_outputs = n_outputs

        # Create Network
        self.network = Network(n_inputs, n_hidden_layers, n_outputs)
    
    def train(self, x_data, y_data, l_rate=0.6, n_epochs=500):

        for epoch in range(n_epochs):
            # Zip the input and output, 
            # so that the input and output of the same datapoint are accessible in the for loop
            for (x, y) in zip(x_data, y_data):
                # Perform a forward pass, calculating the Predicted Y using the existing Network
                self.forward_pass(x)
                
                # Generate the correct Y
                target = np.zeros(self.n_outputs, dtype=np.int)
                target[y] = 1

                # Perform a backward pass, Storing the desired changes of the weights of the network,
                # such that if this input and output was to be forward_pass-ed again, it would
                # predict a value closer to the correct Y
                self.backward_pass(target)

                # Actually perform the weight changes recommended by the backward_pass
                self.update_weights(x, l_rate)

    def forward_pass(self, x):
        # Fills Output of all nodes
        inputs = list(x)

        for layer in self.network.layers:
            for node in layer:
                activation = 0.0
                # Zip node weight and inputs, 
                # and add w * i from all input nodes to the activation
                for (w, i) in zip(node.weight, inputs):
                    activation += w * i
                # The output of the node will be the activation,
                # aka the weighted inputs, of a node ran through a sigmoid function
                node.output = self.sigmoid(activation)
                # Add this value to the inputs.
                # This value will not yet be used in this layer, only for the next layer.
                inputs.append(node.output)
            # Remove the first few elements from inputs, 
            # so that inputs now becomes the inputs of the next layer.
            del inputs[:-len(layer)]
        # Return value is only used in predict function.
        return inputs
    
    def backward_pass(self, target):
        # Fills Delta of all nodes

        # Output Layer
        for i_node, node in enumerate(self.network.layers[-1]):
            # Compare target output with predicted output and get the error margin
            error = target[i_node] - node.output
            # The delta, aka how much the weight should change, 
            # is this error * the derivative of the output node's output.
            node.delta = error * self.derivative(node.output)

        # Hidden Layers
        for i_layer in reversed(range(len(self.network.layers) - 1)):
            # Iterate through layers back to front
            layer = self.network.layers[i_layer]

            for i_node, node in enumerate(layer):
                error = 0.0
                # Use the deltas of the nodes in the layer after this one with the weight
                # connected to the node currently being checked to generate the error for
                # the node currently being checked.
                for next_node in self.network.layers[i_layer + 1]:
                    error += next_node.weight[i_node] * next_node.delta
                # The delta, again, is the error times the derivative of the node's output.
                node.delta = error * self.derivative(node.output)

    def update_weights(self, inputs, l_rate=0.3):
        # Because we only update `inputs` after using the variable, 
        # on the first iteration the value given as a parameter is used.
        # This is done on purpose
        for layer in self.network.layers:
            # The new weight is the current weight plus:
            # the learning rate * the delta (which can be negative or positive) * the input
            for node in layer:
                node.weight = [w + l_rate * node.delta * inputs[i] for i, w in enumerate(node.weight)]
            # We don't need to update this input if the loop is about to terminate. 
            # This if statement is slightly more efficient than not having it.
            if layer != self.network.layers[-1]:
                # Update input for the next layer
                inputs = [node.output for node in layer]

    def predict(self, x):

        # get a zero list which will be filled
        y_predict = np.zeros(len(x), dtype=np.int)
        
        # Iterate over text inputs
        for i, _x in enumerate(x):
            # Get output of forward pass
            output = self.forward_pass(_x)
            # Get highest probability class
            y_predict[i] = np.argmax(output)

        return y_predict

    def sigmoid(self, x):
        # Returns y from a sigmoid function
        return 1.0 / (1.0 + exp(-x))

    def derivative(self, x):
        # Higher if further away from 0 or 1.
        return x * (1.0 - x)

def main():
    # This main function is a modified variant of the one that can be found at https://github.com/ankonzoid/NN-from-scratch
    # That library helped me grasp how Neural Networks can be implemented.

    t = time.time()

    # Edit The following block of values for different results
    filename = "seeds_dataset.csv"
    n_hidden_layers = [5]
    n_outputs = 3
    l_rate = 0.5
    n_epochs = 1000
    n_folds = 4

    # Get inputs and outputs. Also normalize inputs
    X, y = utils.read_csv(filename)
    utils.normalize(X)
    # Get Number of input/output pairs, 
    # and the amount of features each data set has.
    N, features = X.shape

    # Get a range from 0 to N
    idx_all = np.arange(0, N)
    # Gets idx_folds randomly filled arrays, each N / n_folds long
    idx_folds = utils.crossval_folds(N, n_folds, seed=1) 

    # Variables for training and testing scores
    acc_train, acc_test = list(), list()
    print("Training and cross-validating...")
    for i, idx_test in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_test)
        
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]

        # Build neural network classifier model and train
        model = NeuralNetwork(features, n_hidden_layers, n_outputs)
        model.train(X_train, y_train, l_rate=l_rate, n_epochs=n_epochs)

        # Make predictions for training and test data
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==y_train_predict)/len(y_train))
        acc_test.append(100*np.sum(y_test==y_test_predict)/len(y_test))

        # Print cross-validation result
        print(" Fold {}/{}: train acc = {:.2f}%, test acc = {:.2f}% (n_train = {}, n_test = {})".format(i+1, n_folds, acc_train[-1], acc_test[-1], len(X_train), len(X_test)))

    print("Learning Rate: {} with {} Epochs after {:.2f}s.".format(l_rate, n_epochs, time.time() - t))
    print("Epochs: {}".format(n_epochs))
    print("Avg train acc = {:.2f}%".format(sum(acc_train)/float(len(acc_train))))
    print("Avg test acc = {:.2f}%".format(sum(acc_test)/float(len(acc_test))))

if __name__ == "__main__":
    main()
