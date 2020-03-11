from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Data
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y_data = [
    [0],
    [1],
    [1],
    [0]
]

x_data = np.array(x_data)
y_data = np.array(y_data)

# build the model
model = keras.Sequential()

model.add(keras.layers.Dense(4, activation="sigmoid", input_shape=(2,)))
model.add(keras.layers.Dense(6, activation="sigmoid"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

optimize = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=optimize, loss="binary_crossentropy ", metrics=['accuracy'])

model.summary()

model.fit(x_data, y_data, epochs=50000)

predict = model.predict(x_data)
print(predict)

x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)

intensity = []
for i in x:
    input_line = []
    for j in y:
        inp = [i, j]
        input_line.append(inp)
    final_line = []
    pred = model.predict(input_line)
    for p in pred:
        final_line.append(p[0])
    intensity.append(final_line)

# setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

# now just plug the data into pcolormesh, it's that easy!
plt.pcolormesh(x, y, intensity)
plt.colorbar()  # need a colorbar to show the intensity scale
plt.show()  # boom

