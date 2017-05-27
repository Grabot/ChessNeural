
import numpy as np
import os


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1. - y)


def dsigmoid_unsigmoided(y):
    z = sigmoid(y)
    return dsigmoid(z)


def linear(x):
    return x


def dlinear(y):
    return 1


class MLP_Layer(object):
    """
    For now this just holds the data of each layer (it could have some methods).
    """
    def __init__(self, activation, weight_after, bias, activation_function, derivative_activation_function):
        self.activation = activation
        self.raw_activation = np.empty(activation.shape)  # without activation function
        self.weight_after = weight_after
        self.bias = 0 if bias is None else bias  # default bias is 0
        self.delta_weight = np.empty(weight_after.shape) if weight_after is not None else 0
        self.delta_bias = np.empty(activation.shape)
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function


class MLP_NeuralNetwork(object):
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        """
        :param inputNodes: number of input neurons
        :param hiddenNodes: array indicating number of nodes for x hidden layer neurons
        :param outputNodes: number of output neurons
        """
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # set up array of 1s for activations
        self.ai = np.ones(shape=(self.inputNodes,))

        self.ah = []
        for i in range(0, len(hiddenNodes)):
            self.ah.append(np.empty(shape=(self.hiddenNodes[i],)))

        self.ao = np.empty(shape=(self.outputNodes,))

        # create randomized weights
        self.wi = np.random.randn(self.hiddenNodes[0], self.inputNodes)

        self.wh = []
        for i in range(0, len(hiddenNodes)-1):
            self.wh.append(np.random.randn(self.hiddenNodes[i+1], self.hiddenNodes[i]))

        self.wo = np.random.randn(self.outputNodes, self.hiddenNodes[len(hiddenNodes)-1])

        # create arrays of 0 for changes
        self.ci = np.zeros((self.hiddenNodes[0], self.inputNodes))

        self.ch = []
        for i in range(0, len(hiddenNodes)-1):
            self.ch.append(np.zeros((self.hiddenNodes[i+1], self.hiddenNodes[i])))

        self.co = np.zeros((self.outputNodes, self.hiddenNodes[len(hiddenNodes)-1]))
        
        # combine all the layers
        self.layers = [MLP_Layer(self.ai, self.wi, self.ai * 0, None, None)]
        for a, w, c in zip(self.ah, self.wh, self.ch):
            self.layers.append(MLP_Layer(a, w, a * 0, sigmoid, dsigmoid_unsigmoided))
        self.layers += [MLP_Layer(self.ah[-1], self.wo, np.random.randn(*self.ah[-1].shape), sigmoid, dsigmoid_unsigmoided),
                        MLP_Layer(self.ao, None, np.random.randn(*self.ao.shape), linear, dlinear)]
        # for layer in self.layers:  # todo
        #     print('{0:}  {1:}'.format(layer.activation.shape, layer.weight_after.T.shape if layer.weight_after is not None else layer.weight_after))

    def feedForward(self, inputs):

        if len(inputs) != self.inputNodes:
            raise ValueError('Wrong number of inputs!')

        # input activations
        self.layers[0].activation[:] = inputs
        
        # do all the forward propagating uniformly for all the layers
        for k in range(1, len(self.layers)):
            # print(">>", self.layers[k].activation.shape, self.layers[k - 1].weight_after.shape, self.layers[k - 1].activation.shape)
            # print(self.layers[k - 1].weight_after.dot(self.layers[k - 1].activation).shape, self.layers[k].bias.shape)
            self.layers[k].raw_activation[:] = self.layers[k - 1].weight_after.dot(self.layers[k - 1].activation) + self.layers[k].bias
            self.layers[k].activation[:] = self.layers[k].activation_function(self.layers[k].raw_activation[:])
        """ loop code:
        # first hidden activations
        for j in range(self.hiddenNodes[0]):
            sum = 0.0
            for i in range(self.inputNodes):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[0][j] = sigmoid(sum)

        # second and further hidden activations
        for hid in range(1, len(self.hiddenNodes)):
            for j in range(self.hiddenNodes[hid]):
                sum = 0.0
                for i in range(self.hiddenNodes[hid-1]):
                    sum += self.ah[hid-1][i] * self.wh[hid-1][i][j]
                self.ah[hid][j] = sigmoid(sum)

        # output activations
        for k in range(self.outputNodes):
            sum = 0.0
            for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
                sum += self.ah[len(self.hiddenNodes)-1][j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)
        """
        return self.layers[-1].activation

    def backPropagate(self, targets, N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        
        Information at http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation
        """
        
        if os.environ.get('DEBUGLOG', False):
            print('#' * 32)
            for layer in self.layers:
                print('A: ' + '  '.join('{0:8.3f}'.format(act) for act in layer.activation))
                print('  B: ' + '  '.join('{0:8.3f}'.format(act) for act in layer.bias))
                if layer.weight_after is not None:
                    for row in layer.weight_after:
                        print('  W: ' + '  '.join('{0:8.3f}'.format(w) for w in row))
        
        if len(targets) != self.outputNodes:
            raise ValueError('Wrong number of targets')

        # Get the deltas of the output layer
        dcost = self.layers[-1].activation - targets  # (this is specific to quadratic cost function)
        delta = - self.layers[-1].derivative_activation_function(self.layers[-1].raw_activation) * dcost
        
        aw = np.mean(tuple(l.weight_after.mean() if l.weight_after is not None else 0 for l in self.layers))  # todo
        # print('avg weight: {}'.format(aw))
        if np.isnan(aw): exit(666)
        
        # todo: there is probably some easier mathematical operation to avoid this loop
        # for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
        #     errors = deltas * self.layers[-2].weight_after[:, j]
        
        # Do all the back propagating uniformly for all the layers
        for k in range(len(self.layers) - 1, 0, -1):
            pre_layer, post_layer = self.layers[k - 1], self.layers[k]
            # print('layers {0:} <- {1:}'.format(pre_layer.activation.shape[0], post_layer.activation.shape[0]))
            #todo: do weight update
            # print(post_layer.derivative_activation_function)
            # print(pre_layer.activation.shape)
            # print(pre_layer.weight_after.T.shape)
            # print(deltas.shape)
            
            # Calculate the changes in weights and biases
            post_layer.delta_bias[:] = N * delta
            # Update the weights in the network (from input activations and output deltas)
            # print(pre_layer.weight_after.shape)
            # print(deltas.shape)
            # print(pre_layer.activation.shape)
            # print(np.outer(deltas, pre_layer.activation).shape)
            # print(np.outer(pre_layer.activation, deltas).shape)
            pre_layer.delta_weight[:, :] = N * np.outer(delta, pre_layer.activation)
            
            if os.environ.get('DEBUGLOG', False):
                print('    D: ' + '  '.join('{0:8.3f}'.format(d) for d in delta))
                print('   DB: ' + '  '.join('{0:8.3f}'.format(d) for d in post_layer.delta_bias))
                for row in pre_layer.delta_weight:
                    print('   DW: ' + '  '.join('{0:8.3f}'.format(w) for w in row))
           
            # Compute the deltas for the previous layer from those of the current one
            delta = post_layer.derivative_activation_function(pre_layer.raw_activation) * \
                pre_layer.weight_after.T.dot(delta)
            # todo: last delta is useless, no more updates

            # Update the biases in the network
            post_layer.bias += post_layer.delta_bias  # could actually be updated immediately
            # Update the weights in the network (from input activations and output deltas)
            # print(pre_layer.weight_after.shape)
            # print(deltas.shape)
            # print(pre_layer.activation.shape)
            # print(np.outer(deltas, pre_layer.activation).shape)
            # print(np.outer(pre_layer.activation, deltas).shape)
            pre_layer.weight_after += pre_layer.delta_weight
            
            ##for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
            ##    errors = deltas * self.layers[k - 1].weight_after[:, j]
            # errors += deltas * self.layers[k - 1].weight_after[k, :]
            # hidden_deltas[len(self.hiddenNodes)-1][j] = dsigmoid(self.ah[len(self.hiddenNodes)-1][j]) * error
            ##    deltas = self.layers[k].activation_function(self.layers[k].activation) * errors
            # print("^^", self.layers[k].activation.shape, self.layers[k - 1].weight_after.shape, self.layers[k - 1].activation.shape)
            # self.layers[k].activation[:] = sigmoid(
            #     self.layers[k - 1].weight_after.dot(
            #         self.layers[k - 1].activation
            #     )
            # )
        return self.cost_function(self.layers[-1].activation, targets)
        #todo: the rest is never used
        
        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.outputNodes
        for k in range(self.outputNodes):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = []
        for hid in range(0, len(self.hiddenNodes)):
            hidden_deltas.append([0.0] * self.hiddenNodes[hid])

        # calculate error terms for first hidden layer
        # delta tells you which direction to change the weights
        hidden_deltas[len(self.hiddenNodes)-1] = [0.0] * self.hiddenNodes[len(self.hiddenNodes)-1]
        for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
            error = 0.0
            for k in range(self.outputNodes):
                error += output_deltas[k] * self.wo.T[j][k]
            hidden_deltas[len(self.hiddenNodes)-1][j] = dsigmoid(self.ah[len(self.hiddenNodes)-1][j]) * error

        for hid in range(len(self.hiddenNodes), 1, -1):
            # calculate error terms for all hidden layers except first
            # delta tells you which direction to change the weights
            hidden_deltas[hid-2] = [0.0] * self.hiddenNodes[hid-2]
            for j in range(self.hiddenNodes[hid-2]):
                error = 0.0
                for k in range(self.hiddenNodes[hid-1]):
                    error += hidden_deltas[hid-1][k] * self.wh[hid-2].T[j][k]
                hidden_deltas[hid-2][j] = dsigmoid(self.ah[hid-2][j]) * error


        # update the weights connecting second hidden layer to output
        for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
            for k in range(self.outputNodes):
                change = output_deltas[k] * self.ah[len(self.hiddenNodes)-1][j]
                self.wo[k][j] -= N * change + self.co.T[j][k]
                self.co[k][j] = change

        # update the weights connecting the hidden layers
        for hid in range(len(self.hiddenNodes), 1, -1):
            for j in range(self.hiddenNodes[hid-2]):
                for k in range(self.hiddenNodes[hid-1]):
                    change = hidden_deltas[hid-1][k] * self.ah[hid-2][j]
                    self.wh[hid-2][k][j] -= N * change + self.ch[hid-2].T[j][k]
                    self.ch[hid-2][k][j] = change

        # update the weights connecting input to hidden
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes[0]):
                change = hidden_deltas[0][j] * self.ai[i]
                self.wi[j][i] -= N * change + self.ci.T[i][j]
                self.ci[j][i] = change
        
        return self.cost_function(self.ao, targets)
        
    def cost_function(self, output, target):
        # calculate error
        error = 0.0
        for k in range(len(target)):
            error += 0.5 * (target[k] - output[k]) ** 2
        return error

    def train(self, input_series, target_series, iterations=10000, N=0.01):
        # N: learning rate
        assert input_series.shape[0] == target_series.shape[0]
        print('error %8.5f  progress %5d of %5d' % (  # todo: suspicious values
            self.cost_function(self.feedForward(input_series[0]), target_series[0]), 0, iterations))
        for i in range(iterations):
            error = 0.0
            for p in range(input_series.shape[0]):
                inputs = input_series[p, :]
                targets = target_series[p, :]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)
            if i > 0 and i % 1 == 0:
                print('error %8.5f  progress %5d of %5d' % (error, i, iterations))

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p[0]))

        return predictions

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        # for p in patterns:
        #     print(p[0], '->', self.feedForward(p[0]))

    def giveInput(self, input):
        return self.feedForward(input)[0]


