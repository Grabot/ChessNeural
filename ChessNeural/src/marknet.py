
from sys import stdout
from numpy import zeros, copy, newaxis, array, exp, linspace, isfinite, arange, ones
from numpy.random import randn, seed, shuffle


def lin(x):
    return x


def dlin(x):
    return 1


def leakReLU(x):
    return deriv_leakReLU(x) * x


def deriv_leakReLU(x):
    return 0.1 + (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def dsigmoid(x):
    q = sigmoid(x)
    return q * (1 - q)


def quad_cost(actual, expected):
    return ((actual - expected)**2).sum()


def dquad_cost(actual, expected):
    return actual - expected



class Initializer(object):
    def generate(self, shape):
        raise NotImplementedError("Initializers should implement .generate(shape)")


class InitConst(Initializer):
    def __init__(self, val):
        self.val = val
    
    def generate(self, shape):
        return ones(shape) * self.val


class InitNormalRandom(Initializer):
    def generate(self, shape):
        return randn(*shape)


class RescaleWrapper(Initializer):
    def __init__(self, base_initializer):
        self.base_initializer = base_initializer
    
    def generate(self, shape):
        # I think this is called He rescaling but I'm not sure
        return self.base_initializer.generate(shape) * (2 * shape[-1])**-0.5


class Network(object):
    def __init__(self, input_layer, learning_rate=0.01, goal_learning_rate=None, max_epoch_count=100000, stop_at_train_cost=1e-4,
            stop_at_train_test_ratio=3., test_fraction=0.3):
        """
        :param input_layer: The first layer in the network
        :param learning_rate: The initial learning rate
        :param goal_learning_rate: The final learning rate (if not set, equal to initial for a constant learning rate)
        """
        assert 0 <= test_fraction < 1
        self.input_layers = []
        self.output_layers = []
        self.add_input(input_layer)
        self.initial_learning_rate = learning_rate
        self.goal_learning_rate = goal_learning_rate
        self.max_epoch_count = max_epoch_count
        self.stop_at_train_cost = stop_at_train_cost
        self.stop_at_train_test_ratio = stop_at_train_test_ratio
        self.test_fraction = test_fraction
        if (self.goal_learning_rate is None):
            self.goal_learning_rate = learning_rate
        for layer in self.find_ends(input_layer):
            self.add_output(layer)
    
    @staticmethod
    def find_ends(layer):
        if layer.next:
            for nxt in layer.next:
                for res in Network.find_ends(nxt):  # `yield from` in py3.3
                    yield res
        else:
            yield layer
    
    def add_input(self, input_layer):
        assert isinstance(input_layer, InputLayer)
        self.input_layers.append(input_layer)

    def add_output(self, output_layer):
        assert isinstance(output_layer, OutputLayer), str(type(output_layer))
        self.output_layers.append(output_layer)
    
    def execute_for(self, data):
        for input in self.input_layers:
            input.forward(data)
        return tuple(out.activation for out in self.output_layers)
    
    def learn_from(self, expected):
        for output in self.output_layers:
            output.delta_from_expectation(expected, learning_rate=self.initial_learning_rate)
    
    def __str__(self):
        return "network: [{0:}]".format(", ".join(str(layer) for layer in self.input_layers))

    def get_cost(self, expected):
        return sum(layer.get_cost(expected) for layer in self.output_layers)
    
    def flat_layers(self, layers=None):
        if layers is None:
            layers = self.input_layers
        for layer in layers:
            yield layer
            for sub in self.flat_layers(layer.next):
                yield sub
    
    def train(self, input, output, print_every_epochs=10):
        assert len(input.shape) == 2
        assert len(output.shape) in {1, 2}
        assert input.shape[0] == output.shape[0], 'mismatch between input and output shape'
        if len(output.shape) == 1:
            output = output.reshape(output.shape + (1,))
        
        train_sample_count = int(round(self.test_fraction * output.shape[0]))
        test_sample_count = output.shape[0] - train_sample_count
        learning_rates = linspace(self.initial_learning_rate, self.goal_learning_rate, self.max_epoch_count)
        
        stdout.write('{0:>5s} {1:>10s} {2:>10s} {3:>10s}\n'.format('epoch', 'train', 'test', 'ratio'))
        
        # Epochs
        train_costs = []
        test_costs = []
        stop_condition_streak = 0
        for epoch in range(self.max_epoch_count):
            self.initial_learning_rate = learning_rates[epoch]
            epoch_train_cost = 0.
            epoch_test_cost = 0.
            # Train samples
            for inp, outp in zip(input[:train_sample_count], output[:train_sample_count]):
                self.execute_for(inp)
                self.learn_from(outp)
                epoch_train_cost += self.get_cost(outp)
            train_costs.append(epoch_train_cost / train_sample_count)
            # Test samples
            for inp, outp in zip(input[train_sample_count:], output[train_sample_count:]):
                self.execute_for(inp)
                epoch_test_cost += self.get_cost(outp)
            test_costs.append(epoch_test_cost / test_sample_count)
            # Stop conditions
            if not isfinite(train_costs[-1]):
                break
            if train_costs[-1] < self.stop_at_train_cost:
                # if stop_condition_streak >= 5:
                #     print('train cost has reached it\'s goal')
                stop_condition_streak += 1
            elif epoch_train_cost / epoch_test_cost > self.stop_at_train_test_ratio:
                stop_condition_streak += 1
            else:
                stop_condition_streak = 0
            # Printing
            if epoch % print_every_epochs == 0 or stop_condition_streak >= 5 or epoch == self.max_epoch_count - 1:
                stdout.write('{0:5d} {1:10.5f} {2:10.5f} {3:10.5f}\n'.format(epoch, train_costs[-1],
                    test_costs[-1], train_costs[-1] / test_costs[-1]))
            if stop_condition_streak >= 5:
                break
        
        return train_costs, test_costs

    def plot_progress(self, trainc, testc):
        if not trainc:
            return
        from matplotlib.pyplot import subplots, show  # import here so it's not required for the rest of the code
        fig, ax = subplots(tight_layout=True)
        ax.plot(trainc, color='green', label='train error')
        ax.plot(testc, color='red', label='test error')
        ax.set_yscale('log')
        ax.set_title("XOR cost progression")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (log)')
        ax.legend(loc='upper right')
        show()
        

class Layer(object):
    def __init__(self, size, activf=lin, deriv_activf=dlin):
        self.previous = []
        self.next = []
        self.size = size
        self.activf = activf
        self.deriv_activf = deriv_activf
        self.raw_input = zeros((self.size,))
        self.activation = zeros(self.size)
        
    def link(self, next_layer):
        # Note that this may happen last-to-first
        self.next.append(next_layer)
        next_layer.previous.append(self)
        return self
    
    def forward(self, input):
        raise NotImplementedError()
    
    def backward(self, back):
        raise NotImplementedError()
    
    def __str__(self):
        return ("{0:s}%{1:d} -> [{2:s}]".format(
            self.__class__.__name__, self.size,
            ", ".join(str(nxt) for nxt in self.next)
        ))
    
    def get_activation(self):
        return self.activation
    
    def needs_deltas(self):
        # This is a bit ugly, but too much refactoring to remove the need
        return True
    

class DenseLayer(Layer):
    def __init__(self, size, activf=lin, deriv_activf=dlin, bias_initializer=InitConst(0),
            weight_initializer=RescaleWrapper(InitNormalRandom())):
        super(DenseLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
        self.bias = bias_initializer.generate(self.size)
        self._weight_initializer = weight_initializer
        self.weight_before = None
        self.delta = zeros(self.size)
    
    def forward(self, input):
        # Lazy-initialize the weights, because at creation/link time we may not know the size of previous layer
        if self.weight_before is None:
            self.weight_before = self._weight_initializer.generate((input.shape[0], self.size))
        assert self.weight_before.shape[0] == input.shape[0]
        # Weights
        self.raw_input[:] = self.weight_before.T.dot(input)
        # Biasses
        self.raw_input += self.bias
        # Activation
        self.activation[:] = self.activf(self.raw_input)
        # Pass on to next layer
        for nxt in self.next:
            nxt.forward(self.activation)
    
    def backward(self, data, learning_rate):
        # Calculate delta by multiplying by derivative of activation
        self.delta[:] = data * self.deriv_activf(self.raw_input)
        if any(prev.needs_deltas() for prev in self.previous):
            # Calculate the correction input for the previous layer (delta before multiplication by activation)
            prev_delta = self.weight_before.dot(self.delta[:])
            for prev in self.previous:
                prev.backward(prev_delta, learning_rate=learning_rate)
        # Update biasses
        self.bias -= learning_rate * self.delta
        # Update weights (input activation and output delta) (scale by number of previous neurons)
        for prev in self.previous:
            ad_grid = prev.get_activation()[:, newaxis].dot(self.delta[newaxis, :])
            self.weight_before -= ad_grid * (float(learning_rate) / len(self.previous))


class OutputLayer(DenseLayer):
    def __init__(self, size, activf=lin, deriv_activf=dlin, costf=quad_cost, deriv_costf=dquad_cost, bias_initializer=InitConst(0),
            weight_initializer=RescaleWrapper(InitNormalRandom())):
        super(OutputLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf, bias_initializer=bias_initializer,
            weight_initializer=weight_initializer)
        self.costf = costf
        self.deriv_costf = deriv_costf
    
    def forward(self, input):
        super(OutputLayer, self).forward(input)
        assert not self.next
        return self.activation
    
    def __str__(self):
        return ("{0:s}%{1:d}".format(
            self.__class__.__name__, self.size))

    def delta_from_expectation(self, expected, learning_rate):
        self.backward(self.deriv_costf(self.activation, expected), learning_rate=learning_rate)

    def get_cost(self, expected):
        return self.costf(self.activation, expected)


class InputLayer(Layer):
    def forward(self, input):
        self.activation = self.raw_input = copy(input)
        for nxt in self.next:
            nxt.forward(input)
    
    def backward(self, data, learning_rate):
        pass
    
    def needs_deltas(self):
        return False


if __name__ == '__main__':
    seed(123456789)
    
    nn = Network(
        InputLayer(2).link(
        DenseLayer(5, activf=leakReLU, deriv_activf=deriv_leakReLU, weight_initializer=InitNormalRandom()).link(
        ## DenseLayer(16, activf=leakReLU, deriv_activf=deriv_leakReLU).link(
        OutputLayer(1, activf=leakReLU, deriv_activf=deriv_leakReLU, bias_initializer=InitConst(0.5)))),
        # OutputLayer(1, activf=sigmoid, deriv_activf=dsigmoid))),
        learning_rate=0.01,
        goal_learning_rate=0.0001,
        max_epoch_count=10000,
        stop_at_train_cost=1e-4,
        stop_at_train_test_ratio=3,
        test_fraction=0.4
    )
    
    # XOR test data
    sample = arange(0, 40) % 4
    shuffle(sample)
    input = array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])[sample, :]
    output = array([
        0,
        1,
        1,
        0,
    ])[sample]
    
    trainc, testc = nn.train(input, output)
    
    nn.plot_progress(trainc, testc)
    
#todo: why does sigmoid not perform well for classification output layer?
#todo: dropout layer


