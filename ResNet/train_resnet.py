# -*- coding: utf-8 -*-

from ResNet import ResNet50
from read_hdf5 import read_hdf5
import os, sys, h5py
from keras.models import model_from_json
from math import ceil
from datetime import datetime
from LossHistory import LossHistory

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

def train_resnet(
        new_model = False, 
        batch_size = 32, 
        epochs = 20, 
        validation = True
        ):
    
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
    
    steps_per_epoch = int(ceil(n_train / batch_size))
    validation_steps = int(ceil(n_test / batch_size))

    history = LossHistory('epoch_loss.log', 'batch_loss.log')
    
    try:
        model.fit_generator(
                read_hdf5(
                        hdf5_file, 
                        batch_size = batch_size,
                        epochs = epochs
                        ), 
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                verbose = 1, 
                callbacks = [history],
                validation_data = read_hdf5(
                        hdf5_file, 
                        dataset = "test", 
                        batch_size = batch_size,
                        epochs = epochs
                        ),
                validation_steps = validation_steps
                )
        
        if validation:
            eva_data = hdf5_file["val_img"][...]
            eva_labels = hdf5_file["val_labels"][...]
            print("Validation data shape: {}".format(str(eva_data.shape)))
            print("validation labels shape: {}".format(str(eva_labels.shape)))
            preds = model.evaluate(eva_data, eva_labels)
            print("Validation loss: {}".format(preds[0]))
            print("Validation accuracy: {}".format(preds[1]))
        
        hdf5_file.close()
        model_to_json = model.to_json()
        with open("resModel.json", "w") as f:
            f.write(model_to_json)
        print("Model saved to resModel.json!")
        model.save_weights("modelWeights.h5")
        print("Model weights saved to modelWeights.h5")

    except StopIteration:
        hdf5_file.close()
        model_to_json = model.to_json()
        with open("resModel.json", "w") as f:
            f.write(model_to_json)
        print("Model saved to resModel.json!")
        model.save_weights("modelWeights.h5")
        print("Model weights saved to modelWeights.h5")
               
    print("Training time: ", datetime.now() - start_time)
    
def main():
    try:
        new_model = bool(int(sys.argv[1]))
    except IndexError:
        new_model = True
        
    try:
        batch_size = int(sys.argv[2])
    except IndexError:
        batch_size = 32
        
    try:
        epochs = int(sys.argv[3])
    except IndexError:
        epochs = 60
        
    try:
        validation = bool(int(sys.argv[4]))
    except IndexError:
        validation = True
    
    train_resnet(new_model, batch_size, epochs, validation)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
            