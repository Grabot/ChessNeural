
from numpy import array, random
from src.NeuralNetwork import MLP_NeuralNetwork


random.seed(1275464)

inp = array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
outp = array([
    [0],
    [1],
    [1],
    [0],
])
neuro = MLP_NeuralNetwork(2, [3], 1)
neuro.train(inp, outp, iterations=10, N=0.01)
for row in inp:
    print('{1:.0f} xor {2:.0f} = {0:.2f}'.format(neuro.predict([[row]])[0][0], row[0], row[1]))


