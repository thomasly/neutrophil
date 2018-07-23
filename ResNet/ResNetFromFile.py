##

# ResNetFromFile.py
# load model from json
# load weights from HDF5

##

from keras.models import model_from_json
import os
import load_data as ld
from numpy import squeeze

def load_model_from_json(model_path=None, weights_path=None):
	"""
	load dataset and weights from file

	input:
	model_path path to the model file, should be json format
	weights_path path to the weights file, should be HDF5 format

	output:
	Keras model
	"""

	# default model path
	if model_path is None:
		model_path = os.path.abspath(".") + os.sep + "resModel.json"

	# default weights path
	if weights_path is None:
		weights_path = os.path.abspath(".") + os.sep + "modelWeights.h5"

	# read json model file
	json = None
	with open(model_path, "r") as f:
		json = f.read()

	# load model
	model = model_from_json(json)
	# add weights to the model
	model.load_weights(weights_path)

	return model


def test_loaded_model():

	model = load_model_from_json()
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	_, X_test, _, Y_test = ld.load_data(500, 500)
	X_test = X_test / 255
	Y_test = squeeze(Y_test)

	preds = model.evaluate(X_test, Y_test)
	print("Loss = " + str(preds[0]))
	print("Test Accuracy = " + str(preds[1]))


if __name__ == '__main__':
	test_loaded_model()


