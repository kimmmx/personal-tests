"""Manually create DNN to learn xor logic."""


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
        inputs (list): Nested list of inputs as inputs[layer][node].
        outputs (list): Nested list of outputs as outputs[layer][node].
        output_gradients (list): Nested list of partials of model output with respect to node output
            as gradients[layer][node].

    """

    def __init__(self, num_inputs, hidden_shape, activation, learn_rate=1e-4, epochs=int(1e4)):
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
        # Rectified linera unit
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
        """Initializes all weights to 1 and all error gradients to None in the format [net][i][j]."""
        for n in xrange(len(self.node_shape) - 1):
            net_weight = []
            net_gradient = []
            for i in xrange(self.node_shape[n]):
                net_weight.append([1 for _ in xrange(self.node_shape[n + 1])])
                net_gradient.append([None for _ in xrange(self.node_shape[n + 1])])
            self.weights.append(net_weight)
            self.weight_gradients.append(net_gradient)

    def init_nodes(self):
        """Initializes all inputs, outputs, and output gradients to None in the format [layer][node]."""
        for nodes_per_layer in self.node_shape:
            self.inputs.append([None for _ in xrange(nodes_per_layer)])
            self.outputs.append([None for _ in xrange(nodes_per_layer)])
            self.output_gradients.append([None for _ in xrange(nodes_per_layer)])

    def calculate_inputs_outputs(self, train_sample):
        """Calculates inptus and outputs based on training sample and current weights.

        Args:
            train_sample (list): List of inputs and target output.

        """
        for layer in xrange(len(self.node_shape)):
            if layer == 0:
                self.outputs[layer] = train_sample[:-1]
                continue
            for j in xrange(self.node_shape[layer]):
                netj = 0
                for i in xrange(self.node_shape[layer-1]):
                    netj += self.outputs[layer-1][i] * self.weights[layer-1][i][j]
                self.inputs[layer][j] = netj
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

    def get_mse(self, target):
        """Calculates error based on target and current output.

        Note:
            Uses the formula E = (0.5)(t - y)^2, or half the actual MSE, for a convenient derivative.

        Args:
            target (float): Target output.

        """
        return 0.5 * (target - self.outputs[-1][0])**2

    def calculate_error_gradients(self, target):
        """Calculates error gradients based on MSE.

            target (float): Desired output of training sample.
        """
        de_dy = (self.outputs[-1][0] - target)
        for net in xrange(len(self.weights)):
            for i in xrange(len(self.weights[net])):
                for j in xrange(len(self.weights[net][i])):
                    dy_doj = self.output_gradients[net+1][j]
                    doj_dwij = self.outputs[net][i]
                    self.weight_gradients[net][i][j] = de_dy * dy_doj * doj_dwij

    def update_weights(self):
        """Updates weights based on current error gradients."""
        for net in xrange(len(self.weights)):
            for i in xrange(len(self.weights[net])):
                for j in xrange(len(self.weights[net][i])):
                    self.weights[net][i][j] -= self.learn_rate * self.weight_gradients[net][i][j]

    def train_sample(self, train_sample):
        """Trains model with training sample.

        Args:
            train_sample (list): List of inputs and target output.

        """
        target = train_sample[-1]
        self.calculate_inputs_outputs(train_sample)
        self.calculate_output_gradients()
        self.calculate_error_gradients(target)
        self.update_weights()

    def train(self, train_set):
        """Trains model with training set.

        Args:
            train_set (list): List of lists of training data.

        """
        for _ in xrange(self.epochs):
            for train_sample in train_set:
                self.train_sample(train_sample)

    def predict(self, test_set):
        """Predicts on test data.

        Args:
            test_set (list): List of lists of test data.

        """
        for test_sample in test_set:
            self.calculate_inputs_outputs(test_sample)
            print test_sample, self.outputs[-1][0]


def main():

    # Define training set with xor logic
    train_set = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

    # Define shape of network
    num_inputs = 2
    hidden_shape = [2]

    # Create DNN
    model = DeepNeuralNet(
        num_inputs=num_inputs,
        hidden_shape=hidden_shape,
        activation='ReLU'
    )

    model.train(train_set)
    model.predict(train_set)


if __name__ == '__main__':
    main()
