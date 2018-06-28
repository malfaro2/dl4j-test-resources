import keras
from keras.models import Sequential
from keras.layers import SeparableConvolution2D as Sep
import keras.backend as K
import numpy as np

base_path = "./"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

input_shape=(3, 4, 5)
n_out = 6
kernel_size = (3, 3)
depth_multiplier = 2
input_dim = input_shape[2]  # if backend == 'tensorflow' else input_shape[0]

depth_weights = np.arange(0, kernel_size[0] * kernel_size[1] * input_dim * depth_multiplier)
depth_weights = depth_weights.reshape((kernel_size[0], kernel_size[1], input_dim, depth_multiplier))

point_weights = np.arange(0, 1 * 1 * (input_dim * depth_multiplier) * n_out)
point_weights = point_weights.reshape((1, 1, input_dim * depth_multiplier, n_out))

bias = np.zeros((n_out))

model = Sequential()
model.add(Sep(n_out, kernel_size, depth_multiplier=depth_multiplier, input_shape=input_shape))

model.set_weights([depth_weights, point_weights, bias])

input = np.ones((1,) + input_shape)
print(model.predict(input))

# output: [[[[112590. 116595. 120600. 124605. 128610. 132615.] [112590. 116595. 120600. 124605. 128610. 132615.]]]], shape 1, 1, 2, 6


model.compile(loss='mse', optimizer='adam')

print("Saving model with single separable convolution layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}sepconv2d_{}_{}.h5".format(base_path, backend, major_version))
