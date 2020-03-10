
"""
This is data from Kaggle's Titanic competition.
https://www.kaggle.com/c/titanic/
"""

import csv
from numpy import array, log
from marknet import Network, InputLayer, DenseLayer, OutputLayer, leakReLU, deriv_leakReLU, InitConst, DropoutLayer


def yield_data():
    with open('titanic_train.csv', 'r') as fh:
        rdr = csv.reader(fh)
        next(rdr)   # skip header
        for parts in rdr:
            cabins = tuple(cabin for cabin in parts[10].strip().split() if cabin)
            cabin_count = len(cabins)
            cabin_letter = cabin_nr = 0
            if cabins:
                cabin_letter = cabins[0][0]
                cabin_nr = float('0' + cabins[0][1:])
            has_age = False
            try:
                age_rescaled = float(parts[5]) / 50
                has_age = True
            except ValueError:
                age_rescaled = 0
            yield [
                int(parts[1]),             # label (survived?)
                int(parts[0]),             # id
                int(parts[2]) == 1,        # first class
                int(parts[2]) == 2,        # second class
                int(parts[2]) == 3,        # third class
                ' Mr.' in parts[3],        # title
                ' Mrs.' in parts[3],       # title
                ' Miss.' in parts[3],      # title
                ' Rev.' in parts[3],       # title
                ' Master.' in parts[3],    # title
                ' Major.' in parts[3],     # title
                ' Dr.' in parts[3],        # title
                'fe' in parts[4],          # gender (mostly included in title, but anyway)
                has_age,                   # is age specified?
                age_rescaled,              # age
                float(parts[6]),           # siblings on board
                float(parts[7]),           # parents on board
                # skip the ticket number, hard to convert to to something numerical and useful
                float(parts[9]) / 150,     # price
                log(1 + float(parts[9])) / 5,  # log price
                cabin_count,               # number of cabins
                cabin_letter == 'A',       # cabin range A
                cabin_letter == 'B',       # cabin range B
                cabin_letter == 'C',       # cabin range C
                cabin_letter == 'D',       # cabin range D
                cabin_letter == 'E',       # cabin range E
                cabin_letter == 'F',       # cabin range F
                cabin_nr,                  # nr of first cabin
                parts[11] == 'Q',          # embarked at Queenstown
                parts[11] == 'C',          # embarked at Cherbourg
                parts[11] == 'S',          # embarked at Southampton
            ]

# Get the data
labels = []
ids = []
data = []
for row in yield_data():
    labels.append(row[0])
    ids.append(row[1])
    data.append(row[2:])
labels = array(labels)
data = array(data, dtype=float)

# Set up the network
nn = Network(
    InputLayer(2).link(
        DenseLayer(40, activf=leakReLU, deriv_activf=deriv_leakReLU).link(
        DropoutLayer(40).link(
        DenseLayer(30, activf=leakReLU, deriv_activf=deriv_leakReLU).link(
        DropoutLayer(30).link(
        DenseLayer(25, activf=leakReLU, deriv_activf=deriv_leakReLU).link(
        DropoutLayer(25).link(
        OutputLayer(1, activf=leakReLU, deriv_activf=deriv_leakReLU, bias_initializer=InitConst(0.5))
    ))))))),
    learning_rate=0.0002,
    goal_learning_rate=0.00001,
    max_epoch_count=5000,
    stop_at_train_cost=1e-4,
    stop_at_train_test_ratio=3,
    test_fraction=0.4
)

# Do training
trainc, testc = nn.train(data, labels)
nn.plot_progress(trainc, testc)


