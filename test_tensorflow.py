# import tensorflow as tf
from keras.models import Model
from keras.layers import MaxPooling2D, Flatten, Dense, Input
import numpy as np


def test():
	X_train = np.random.randn(10, 2, 2, 3)
	Y_train = np.random.randn(10, 1)
	input_shape = (2, 2, 3)
	X_input = Input(input_shape)
	X = MaxPooling2D((1, 1), strides = (1, 1))(X_input)
	X = Flatten()(X)
	X = Dense(1, activation = "sigmoid")(X)
	model = Model(inputs = X_input, outputs = X)
	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
	model.fit(X_train, Y_train, epochs = 1, batch_size = 1)

if __name__ == '__main__':
	test()
