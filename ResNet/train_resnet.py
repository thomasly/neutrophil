# -*- coding: utf-8 -*-

from ResNet import ResNet50
from read_hdf5 import read_hdf5
import os, h5py
from keras.models import model_from_json
from math import ceil
from datetime import datetime

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

def train_resnet(new_model=False, batch_size = 32):
    """
    """
    start_time = datetime.now()
    hdf5_path = os.path.join("..", "data", "76_79_80.hdf5")
    if new_model:
        model = ResNet50()
        
    else:
        model = load_model_from_json()
        
    model.compile(
            optimizer = "adam", 
            loss = "binary_crossentropy", 
            metrics = ["accuracy"]
            )
    
    hdf5_file = h5py.File(hdf5_path, mode = 'r')
    n_train = hdf5_file["train_img"].shape[0]
    n_test = hdf5_file["test_img"].shape[0]
    hdf5_file.close()
    
    steps_per_epoch = int(ceil(n_train / batch_size))
    validation_steps = int(ceil(n_test / batch_size))
    model.fit_generator(
            read_hdf5(hdf5_path, batch_size = batch_size), 
            steps_per_epoch = steps_per_epoch,
            epochs = 20,
            verbose = 1, 
            callbacks = None,
            validation_data = read_hdf5(
                    hdf5_path, 
                    dataset = "test", 
                    batch_size = batch_size
                    ),
            validation_steps = validation_steps
            )
            
    model.save_weights("modelWeights.h5")
    
    print("Training time: ", datetime.now() - start_time)
    
def main():
    train_resnet()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
            