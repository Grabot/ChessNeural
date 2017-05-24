
from numpy import zeros, copy, ones
from numpy.random import randn, seed


seed(1234567)


def lin(x):
    return x


def dlin(x):
    return 1


class Layer(object):
    def __init__(self, size, activf=lin, deriv_activf=dlin):
        self.previous = []
        self.next = []
        self.size = size
        self.activf = activf
        self.deriv_activf = deriv_activf
        
    def link(self, next_layer):
        # note that this may happen last-to-first
        self.next.append(next_layer)
        next_layer.previous.append(self)
        return self
    
    def forward(self, input):
        raise NotImplementedError()
    
    def backward(self, input):
        raise NotImplementedError()
    
    def __str__(self):
        return ("{0:s}%{1:d} -> [{2:s}]".format(
            self.__class__.__name__, self.size,
            ", ".join(str(nxt) for nxt in self.next)
        ))
        

class DenseLayer(Layer):
    def __init__(self, size, activf=lin, deriv_activf=dlin):
        super(DenseLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
        self.raw_input = zeros(self.size,)
        self.activation = zeros(self.size,)
        self.bias = randn(self.size)
        self.weight_before = None

    # def link(self, next_layer):
        # super(DenseLayer, self).link(next_layer)
        # assert len(self.previous) == 1, "{0:d} previous layers instead of 1 for {1:}".format(len(self.previous), self)
        # self.weight_before = randn(self.previous[0].size, self.size)
    
    def forward(self, input):
        print('forward!')
        if self.weight_before is None:
            self.weight_before = randn(input.shape[0], self.size)
        assert self.weight_before.shape[0] == input.shape[0]
        # weights
        self.raw_input = self.weight_before.T.dot(input)
        # biasses
        self.raw_input += self.bias
        # activation
        self.activation = self.activf(self.raw_input)
        # pass on
        for nxt in self.next:
            nxt.forward(input)


class OutputLayer(DenseLayer):
    def forward(self, input):
        super(OutputLayer, self).forward(input)
        assert not self.next
        print("output activation(s): {0:s}".format(", ".join("{0:.4f}".format(a) for a in self.activation)))
    
    def __str__(self):
        return ("{0:s}%{1:d}".format(
            self.__class__.__name__, self.size))
    

class InputLayer(Layer):
    def __init__(self, size, activf=lin, deriv_activf=dlin):
        super(InputLayer, self).__init__(size, activf=activf, deriv_activf=deriv_activf)
        self.raw_input = zeros((self.size,))
    
    def forward(self, input):
        self.raw_input = copy(input)
        for nxt in self.next:
            nxt.forward(input)

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

nn = InputLayer(2).link(
    DenseLayer(10).link(
    DenseLayer(6).link(
    OutputLayer(1))))


print(nn)
nn.forward(ones(2))


# network.forward(ones((3,)))