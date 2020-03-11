from numpy import exp, array, newaxis, empty, sqrt
from numpy.random import randn
import math


def sigmoid(x):
    return 1 / (1 + exp(-x))


# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1.0 - y)


# Simpler, better activation functions
def leakReLU(x):
    return deriv_leakReLU(x) * x


def deriv_leakReLU(x):
    return 0.1 + (x > 0)


# Store the data in a layer class
class MLP_Layer(object):
    def __init__(self, layer_before, size):
        # Use numpy arrays for everything (not Python lists)
        self.previous = layer_before
        self.next = None
        self.size = size
        self.raw_input = empty((self.size,))
        self.activation = empty(self.size)
        self.bias = self.weight_before = None
        if self.previous:
            self.previous.next = self
            self.bias = randn(self.size)
            self.weight_before = sqrt(2 / self.size) * randn(self.previous.size, self.size)
    

class MLP_NeuralNetwork(object):
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        """
        :param inputNodes: number of input neurons
        :param hiddenNodes: array indicating number of nodes for x hidden layer neurons
        :param outputNodes: number of output neurons
        """
        self.inputNodes = inputNodes + 1
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        # Less code repetition with Layer objects
        self.layers = [MLP_Layer(None, inputNodes)]
        for hiddenSize in hiddenNodes:
            self.layers.append(MLP_Layer(self.layers[-1], hiddenSize))
        self.layers.append(MLP_Layer(self.layers[-1], outputNodes))

    def feedForward(self, inputs):

        if len(inputs) != self.inputNodes - 1:
            raise ValueError('Wrong number of inputs!')

        # input activations
        self.layers[0].activation[:] = array(inputs)

        # first hidden activations
        # second and further hidden activations
        # output activations
        for hid in range(1, len(self.hiddenNodes) + 2):
            # Use numpy .dot everywhere instead of nested for loops
            # Get the activations from the previous layer
            input = self.layers[hid - 1].activation
            # Weights
            self.layers[hid].raw_input[:] = self.layers[hid].weight_before.T.dot(input)
            # Biasses
            self.layers[hid].raw_input += self.layers[hid].bias
            # Activation
            self.layers[hid].activation[:] = leakReLU(self.layers[hid].raw_input)
        
        return self.layers[-1].activation
    
    def backPropagate(self, targets, N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        """
        if len(targets) != self.outputNodes:
            raise ValueError('Wrong number of targets')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        # specific for quadratic cost function
        delta = (self.layers[-1].activation - array(targets)) * deriv_leakReLU(self.layers[-1].raw_input)

        for hid in reversed(range(1, len(self.hiddenNodes) + 2)):
            # update the weights connecting second hidden layer to output
            # update the weights connecting the hidden layers
            # update the weights connecting input to hidden
            # use `nexaxis` magic to make a 2D matrix out of 2 1D vectors
            ad_grid = self.layers[hid - 1].activation[:, newaxis].dot(delta[newaxis, :])
            self.layers[hid].weight_before -= ad_grid * N
            
            # also update the biasses
            self.layers[hid].bias -= N * delta
            
            # Calculate delta for previous layer (except the input layer, we don't need that)
            if hid != 1:
                delta = self.layers[hid].weight_before.dot(delta)
        
        return self.error(targets)
        
    def error(self, targets):
        # calculate error
        # specific for quadratic error
        error = ((self.layers[-1].activation - array(targets))**2).sum()
        return error

    def train(self, patterns, iterations=20000, N=0.01):
        # N: learning rate
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)
            if i % 10 == 0:
                print('error %-.5f' % error, " progress", i, "of", iterations)

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[0], '->', self.feedForward(p[0]))

    def give_input(self, input):
        return self.feedForward(input)[0]



