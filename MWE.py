
from numpy import zeros, copy, ones, newaxis, array, logical_xor
from numpy.random import randn, seed


seed(1234567)


def lin(x):
    return x


def dlin(x):
    return 1


def quad_cost(x):
    return (x**2).sum()


def dquad_cost(x):
    return x


class Network(object):
    def __init__(self, input_layer, learning_rate=0.01):
        self.input_layers = []
        self.output_layers = []
        self.add_input(input_layer)
        self.learning_rate = learning_rate
        for layer in self.find_ends(input_layer):
            self.add_output(layer)
    
    @staticmethod
    def find_ends(layer):
        if layer.next:
            for nxt in layer.next:
                for res in Network.find_ends(nxt):  # yield from in py3.3
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
    
    def learn_from(self, expected):
        for output in self.output_layers:
            # print(">>> learn from output " , output)
            output.delta_from_expectation(expected, learning_rate=self.learning_rate)
    
    def __str__(self):
        return "network: [{0:}]".format(", ".join(str(layer) for layer in self.input_layers))

    def get_cost(self):
        return sum(layer.get_cost() for layer in self.output_layers)
    
    def flat_layers(self, layers=None):
        if layers is None:
            layers = self.input_layers
        for layer in layers:
            yield layer
            for sub in self.flat_layers(layer.next):
                yield sub


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
        # note that this may happen last-to-first
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
    def __init__(self, size, activf=lin, deriv_activf=dlin):
        super(DenseLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
        # self.raw_input = zeros(self.size)
        self.bias = randn(self.size)
        self.weight_before = None
        self.delta = zeros(self.size)

    # def link(self, next_layer):
        # super(DenseLayer, self).link(next_layer)
        # assert len(self.previous) == 1, "{0:d} previous layers instead of 1 for {1:}".format(len(self.previous), self)
        # self.weight_before = randn(self.previous[0].size, self.size)
    
    def forward(self, input):
        if self.weight_before is None:
            self.weight_before = randn(input.shape[0], self.size)
        assert self.weight_before.shape[0] == input.shape[0]
        # weights
        self.raw_input[:] = self.weight_before.T.dot(input)
        # biasses
        self.raw_input += self.bias
        # activation
        self.activation[:] = self.activf(self.raw_input)
        # pass on
        for nxt in self.next:
            nxt.forward(self.activation[:])
    
    def backward(self, data, learning_rate):
        print("back", data.shape)
        # calculate delta by multiplying by derivative of activation
        self.delta[:] = data * self.deriv_activf(self.raw_input)
        if any(prev.needs_deltas() for prev in self.previous):
            # Calculate the correction input for the previous layer (delta before multiplication by activation)
            prev_delta = self.weight_before.dot(self.delta[:])
            for prev in self.previous:
                prev.backward(prev_delta, learning_rate=learning_rate)
        # Update biasses
        self.bias += learning_rate * self.delta
        # Update weights (input activation and output delta) (scale by number of previous neurons)
        for prev in self.previous:
            ad_grid = prev.get_activation()[:, newaxis].dot(self.delta[newaxis, :])
            self.weight_before += learning_rate * ad_grid / len(self.previous)

    # def link(self, next_layer):
    #     res = super(DenseLayer, self).link(next_layer)
    #     assert len(self.previous) <= 1
    #     return res


class OutputLayer(DenseLayer):
    def __init__(self, size, activf=lin, deriv_activf=dlin, costf=quad_cost, dcostf=dquad_cost):
        super(OutputLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
        self.costf = costf
        self.dcostf = dcostf
    
    def forward(self, input):
        super(OutputLayer, self).forward(input)
        assert not self.next
        # print("output activation(s): {0:s}".format(", ".join("{0:.4f}".format(a) for a in self.activation)))
    
    def __str__(self):
        return ("{0:s}%{1:d}".format(
            self.__class__.__name__, self.size))

    def delta_from_expectation(self, expected, learning_rate):
        self.backward(self.dcostf(self.activation - expected), learning_rate=learning_rate)

    def get_cost(self):
        return self.costf(self.activation)


class InputLayer(Layer):
    # def __init__(self, size, activf=lin, deriv_activf=dlin):
    #     super(InputLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
    #     self.raw_input = zeros((self.size,))

    def forward(self, input):
        self.activation = self.raw_input = copy(input)
        for nxt in self.next:
            nxt.forward(input)
    
    def backward(self, data, learning_rate):
        # print("backpropagated! data.shape = {}".format(data.shape))  # todo
        pass
    
    def needs_deltas(self):
        return False

# il = InputLayer(2)
# dl = DenseLayer(10)
# sl = DenseLayer(6)
# print(il)
# print(dl)
# print(sl)
# il.link(dl)
# dl.link(sl)
# print(il)
# print(dl)
# print(sl)
# exit()

nn = Network(
    InputLayer(2).link(
    DenseLayer(10).link(
    DenseLayer(6).link(
    OutputLayer(1)))),
    learning_rate= 0.01
)

for k in range(10):
    inp = array([(k // 2) % 2, k % 2])
    outp = array(logical_xor(*inp), dtype=int)
    print(inp, outp)
    nn.execute_for(inp)
    print('cost: {0:.3f}'.format(nn.get_cost()))
    nn.learn_from(outp)

# nn.execute_for(ones(2))
# for layer in nn.flat_layers():
#     print(layer.size, '->', layer.weight_before.shape if hasattr(layer, 'weight_before') else '')
#
# print("cost: {0:.4f}".format(nn.get_cost()))
# nn.learn_from(0.5)
#
# network.forward(ones((3,)))

#todo: dropout layer

