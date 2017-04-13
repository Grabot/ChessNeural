import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1.0 - y)

class MLP_NeuralNetwork(object):
    def __init__(self, inputNodes, hiddenNodes1, hiddenNodes2, outputNodes):
        """
        :param inputNodes: number of input neurons
        :param hidden1: number of hidden neurons
        :param output: number of output neurons
        """
        self.inputNodes = inputNodes + 1
        self.hiddenNodes1 = hiddenNodes1
        self.hiddenNodes2 = hiddenNodes2
        self.outputNodes = outputNodes

        # set up array of 1s for activations
        self.ai = [1.0] * self.inputNodes
        self.ah1 = [1.0] * self.hiddenNodes1
        self.ao = [1.0] * self.outputNodes

        # create randomized weights
        self.wi = np.random.randn(self.inputNodes, self.hiddenNodes1)
        self.wo = np.random.randn(self.hiddenNodes1, self.outputNodes)

        # create arrays of 0 for changes
        self.ci = np.zeros((self.inputNodes, self.hiddenNodes1))
        self.co = np.zeros((self.hiddenNodes1, self.outputNodes))


    def feedForward(self, inputs):

        if len(inputs) != self.inputNodes - 1:
            raise ValueError('Wrong number of inputs!')

            # input activations
        for i in range(self.inputNodes - 1):
            self.ai[i] = inputs[i]

            # hidden activations
        for j in range(self.hiddenNodes1):
            sum = 0.0
            for i in range(self.inputNodes):
                sum += self.ai[i] * self.wi[i][j]
            self.ah1[j] = sigmoid(sum)

            # output activations
        for k in range(self.outputNodes):
            sum = 0.0
            for j in range(self.hiddenNodes1):
                sum += self.ah1[j] * self.wo[j][k]
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

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hiddenNodes1
        for j in range(self.hiddenNodes1):
            error = 0.0
            for k in range(self.outputNodes):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah1[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hiddenNodes1):
            for k in range(self.outputNodes):
                change = output_deltas[k] * self.ah1[j]
                self.wo[j][k] -= N * change + self.co[j][k]
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes1):
                change = hidden_deltas[j] * self.ai[i]
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