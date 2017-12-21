"""Manually create DNN to learn xor logic."""
from tqdm import tqdm
from random import random


class DeepNeuralNet(object):
    """Unbiased, fully-connected DNN for regression.

    This is a rudimentary model that creates a DNN without biases or dropout.
    All inputs must be real. The model trains for a single real output.

    Attributes:
        node_shape (list): List of nodes per layer.
        learn_rate (float): Learning rate for updating weights.
        epochs (int): Number of epochs to iterate over while training.
        phi (function): Activation function.
        dphi_dnetj (function): Derivative of activation function with respect to input.
        weights (list): Nested list of weights as weights[net][i][j].
        weight_gradients (list): Nested list of partials of MSE with respect to weights as gradients[net][i][j].
        biases (list): Nested list of node biases as biases[layer][node].
        bias_gradients (list): Nested list of bias gradients as gradients[layer][node].
        inputs (list): Nested list of inputs as inputs[layer][node].
        outputs (list): Nested list of outputs as outputs[layer][node].
        output_gradients (list): Nested list of partials of model output with respect to node output
            as gradients[layer][node].

    """

    def __init__(self, num_inputs, hidden_shape, activation='ReLU', learn_rate=1e-4, epochs=int(1e4)):
        """ Initializes shape of weights and outputs of DNN.

        Args:
            num_inputs (int): Number of input variables. The network is assumed to have a single output.
            hidden_shape (list): List of integers corresponding to number of nodes in hidden layers.
            activation (str): Name of activation function to use while training.
            learn_rate (float, optional): Learning rate for updating weights. Defaults to 1e-4.
            epochs (int, optional): Number of epochs to iterate over while training.

        """
        self.node_shape = [num_inputs] + [item for item in hidden_shape] + [1]
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.phi = None
        self.dphi_dnetj = None
        self.weights = []
        self.weight_gradients = []
        self.biases = []
        self.bias_gradients = []
        self.inputs = []
        self.outputs = []
        self.output_gradients = []
        self.init_weights()
        self.init_nodes()
        self.get_activation(activation)

    def get_activation(self, function_name):
        """Selects activation function and its derivative.

        Args:
            function_name (str): Name of activation function.

        """
        # Rectified linear unit
        if function_name == 'ReLU':
            def phi(netj):
                return netj if netj > 0 else 0

            def dphi_dnetj(netj):
                return 1 if netj > 0 else 0

        else:
            print "Unknown activation function."
            phi, dphi_dnetj = None, None

        self.phi = phi
        self.dphi_dnetj = dphi_dnetj

    def init_weights(self):
        """Initializes all weights in the format [layer][i][j].

            Note: 'i' corresponds to the current layer's node, and 'j' to the next layer's node.

        """
        for layer in xrange(len(self.node_shape) - 1):
            net_weight = []
            net_gradient = []
            for i in xrange(self.node_shape[layer]):
                net_weight.append([1 for _ in xrange(self.node_shape[layer + 1])])
                net_gradient.append([None for _ in xrange(self.node_shape[layer + 1])])
            self.weights.append(net_weight)
            self.weight_gradients.append(net_gradient)

    def init_nodes(self):
        """Initializes all inputs, outputs, and biases in the format [layer][node]."""
        for nodes_per_layer in self.node_shape:
            self.inputs.append([None for _ in xrange(nodes_per_layer)])
            self.outputs.append([None for _ in xrange(nodes_per_layer)])
            self.output_gradients.append([None for _ in xrange(nodes_per_layer)])
            self.biases.append([2 * random() - 1 for _ in xrange(nodes_per_layer)])
            self.bias_gradients.append([None for _ in xrange(nodes_per_layer)])

    def calculate_inputs_outputs(self, sample, predict=False):
        """Calculates inputs and outputs based on training sample and current weights.

        Args:
            sample (list): List of inputs and target output.
            predict (bool): Indicates whether data includes target output or not.

        """
        for layer in xrange(len(self.node_shape)):
            if layer == 0:
                if not predict:
                    self.outputs[layer] = sample[:-1]
                else:
                    self.outputs[layer] = sample
                continue
            for j in xrange(self.node_shape[layer]):
                netj = self.biases[layer][j]
                for i in xrange(self.node_shape[layer-1]):
                    netj += self.outputs[layer-1][i] * self.weights[layer-1][i][j]
                self.inputs[layer][j] = netj
                if layer == len(self.node_shape) - 1:
                    self.outputs[layer][j] = netj
                else:
                    self.outputs[layer][j] = self.phi(netj)

    def calculate_output_gradients(self):
        """Calculates output gradients based on current weights."""
        # Loop through layers backwards
        for layer in reversed(xrange(len(self.node_shape))):
            if layer == len(self.node_shape) - 1:  # Output layer has gradient 1
                self.output_gradients[-1][0] = 1
                continue
            for i in xrange(self.node_shape[layer]):
                gradient = 0
                for j in xrange(self.node_shape[layer+1]):
                    dnetj_doi = self.weights[layer][i][j]
                    doj_dnetj = self.dphi_dnetj(self.inputs[layer+1][j])
                    dy_doj = self.output_gradients[layer+1][j]
                    gradient += dnetj_doi * doj_dnetj * dy_doj
                self.output_gradients[layer][i] = gradient

    def calculate_weight_gradients(self, target):
        """Calculates error gradients based on error = (0.5)(t - y)^2.

            target (float): Desired output of training sample.
        """
        de_dy = (self.outputs[-1][0] - target)
        for net in xrange(len(self.weights)):
            for i in xrange(len(self.weights[net])):
                for j in xrange(len(self.weights[net][i])):
                    dy_doj = self.output_gradients[net+1][j]
                    doj_dwij = self.outputs[net][i]
                    self.weight_gradients[net][i][j] = de_dy * dy_doj * doj_dwij

    def calculate_bias_gradients(self, target):
        de_dy = (self.outputs[-1][0] - target)
        for layer in xrange(len(self.biases)):
            for node in xrange(len(self.biases[layer])):
                dy_doj = self.output_gradients[layer][node]
                doj_dbj = self.dphi_dnetj(self.inputs[layer][node])
                self.bias_gradients[layer][node] = de_dy * dy_doj * doj_dbj

    def update_weights(self):
        """Updates weights based on current weight gradients."""
        for net in xrange(len(self.weights)):
            for i in xrange(len(self.weights[net])):
                for j in xrange(len(self.weights[net][i])):
                    self.weights[net][i][j] -= self.learn_rate * self.weight_gradients[net][i][j]

    def update_biases(self):
        """Updates biases based on current bias gradients"""
        for layer in xrange(len(self.biases)):
            for node in xrange(len(self.biases[layer])):
                self.biases[layer][node] -= self.learn_rate * self.bias_gradients[layer][node]

    def train_sample(self, train_sample):
        """Trains model with training sample.

        Args:
            train_sample (list): List of inputs and target output.

        """
        target = train_sample[-1]
        self.calculate_inputs_outputs(train_sample)
        self.calculate_output_gradients()
        self.calculate_weight_gradients(target)
        self.calculate_bias_gradients(target)
        self.update_weights()
        self.update_biases()

    def train(self, train_set):
        """Trains model with training set.

        Args:
            train_set (list): List of lists of training data.

        """
        for _ in tqdm(xrange(self.epochs)):
            for train_sample in train_set:
                self.train_sample(train_sample)

    def predict(self, test_set):
        """Predicts on test data.

        Args:
            test_set (list): List of lists of test data.

        """
        for test_sample in test_set:
            self.calculate_inputs_outputs(test_sample, predict=True)
            print test_sample, self.outputs[-1][0]


def main():

    # Define training and test set with xor logic
    train_set = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    test_set = [sample[:2] for sample in train_set]

    # Define shape of network
    num_inputs = 2
    hidden_shape = [2]

    # Create DNN
    model = DeepNeuralNet(
        num_inputs=num_inputs,
        hidden_shape=hidden_shape,
        learn_rate=1e-3,
        epochs=50000
    )

    model.train(train_set)
    model.predict(test_set)
    print model.weights
    print model.biases


if __name__ == '__main__':
    main()
