import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1.0 - y)

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

        # set up array of 1s for activations
        self.ai = [1.0] * self.inputNodes

        self.ah = []
        for i in range(0, len(hiddenNodes)):
            self.ah.append([1.0] * self.hiddenNodes[i])

        self.ao = [1.0] * self.outputNodes

        # create randomized weights
        self.wi = np.random.randn(self.inputNodes, self.hiddenNodes[0])

        self.wh = []
        for i in range(0, len(hiddenNodes)-1):
            self.wh.append(np.random.randn(self.hiddenNodes[i], self.hiddenNodes[i+1]))

        self.wo = np.random.randn(self.hiddenNodes[len(hiddenNodes)-1], self.outputNodes)

        # create arrays of 0 for changes
        self.ci = np.zeros((self.inputNodes, self.hiddenNodes[0]))

        self.ch = []
        for i in range(0, len(hiddenNodes)-1):
            self.ch.append(np.zeros((self.hiddenNodes[i], self.hiddenNodes[i+1])))

        self.co = np.zeros((self.hiddenNodes[len(hiddenNodes)-1], self.outputNodes))


    def feedForward(self, inputs):

        if len(inputs) != self.inputNodes - 1:
            raise ValueError('Wrong number of inputs!')

        # input activations
        for i in range(self.inputNodes - 1):
            self.ai[i] = inputs[i]

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

        return self.ao[:]

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
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[len(self.hiddenNodes)-1][j] = dsigmoid(self.ah[len(self.hiddenNodes)-1][j]) * error

        for hid in range(len(self.hiddenNodes), 1, -1):
            # calculate error terms for all hidden layers except first
            # delta tells you which direction to change the weights
            hidden_deltas[hid-2] = [0.0] * self.hiddenNodes[hid-2]
            for j in range(self.hiddenNodes[hid-2]):
                error = 0.0
                for k in range(self.hiddenNodes[hid-1]):
                    error += hidden_deltas[hid-1][k] * self.wh[hid-2][j][k]
                hidden_deltas[hid-2][j] = dsigmoid(self.ah[hid-2][j]) * error


        # update the weights connecting second hidden layer to output
        for j in range(self.hiddenNodes[len(self.hiddenNodes)-1]):
            for k in range(self.outputNodes):
                change = output_deltas[k] * self.ah[len(self.hiddenNodes)-1][j]
                self.wo[j][k] -= N * change + self.co[j][k]
                self.co[j][k] = change

        # update the weights connecting the hidden layers
        for hid in range(len(self.hiddenNodes), 1, -1):
            for j in range(self.hiddenNodes[hid-2]):
                for k in range(self.hiddenNodes[hid-1]):
                    change = hidden_deltas[hid-1][k] * self.ah[hid-2][j]
                    self.wh[hid-2][j][k] -= N * change + self.ch[hid-2][j][k]
                    self.ch[hid-2][j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes[0]):
                change = hidden_deltas[0][j] * self.ai[i]
                self.wi[i][j] -= N * change + self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2

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
            if i % 1000 == 0:
                print('error %-.5f' % error, " progress", i, "of", iterations)


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
        for p in patterns:
            print(p[0], '->', self.feedForward(p[0]))

    def giveInput(self, input):
        return self.feedForward(input)[0]