##

# ResNetFromFile.py
# load model from json
# load weights from HDF5

##

from keras.models import model_from_json
import os

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
    home_path = os.path.abspath(".")
    if model_path is None:
        model_path = os.path.join(home_path, "resModel.json")

    # default weights path
    if weights_path is None:
        weights_path = os.path.join(home_path, "modelWeights.h5")

    # read json model file
    json = None
    with open(model_path, "r") as f:
        json = f.read()

    # load model
    model = model_from_json(json)
    # add weights to the model
    model.load_weights(weights_path)

    return model



