'''
Filename: train_resnet.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 4, 2018 12:34 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

from ResNet import ResNet50
from read_hdf5 import read_hdf5
import os
import sys
import logging
import tables
from paths import Paths
from keras.models import model_from_json
from keras.callbacks import TensorBoard
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
        new_model=False,
        batch_size=32,
        epochs=20,
        validation=True):
    """
    """
    start_time = datetime.now()
    paths = Paths()
    model_name = "ResNet50"
    hdf5_path = os.path.join("..", "data", "76_79_80_noValidate.hdf5")
    if new_model:
        model = ResNet50()

    else:
        model = load_model_from_json()

    model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )

    hdf5_file = tables.open_file(hdf5_path, mode='r')
    n_train = hdf5_file.root.train_img.shape[0]
    n_test = hdf5_file.root.test_img.shape[0]

    steps_per_epoch = int(ceil(n_train / batch_size))
    validation_steps = int(ceil(n_test / batch_size))

    timestamp = datetime.now().strftime(r"%Y%m%d_%I%M%p")
    tb_log_path = os.path.join(
        paths.logs, '{}_logs_{}'.format(model_name, timestamp))
    os.makedirs(tb_log_path, exist_ok=True)
    os.makedirs(paths.models, exist_ok=True)

    epoch_loss_path = os.path.join(
        paths.logs,
        "{}_epoch_loss_{}.log".format(model_name, timestamp)
    )
    batch_loss_path = os.path.join(
        paths.logs,
        "{}_batch_loss_{}.log".format(model_name, timestamp)
    )
    model_hdf5_path = os.path.join(
        paths.models,
        "{}_{}.hdf5".format(model_name, timestamp)
    )
    history = LossHistory(
        epoch_loss_path,
        batch_loss_path,
        model_hdf5_path
    )
    tensorboard = TensorBoard(log_dir=tb_log_path)

    try:
        model.fit_generator(
                read_hdf5(
                        hdf5_file,
                        batch_size=batch_size,
                        ),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
                callbacks=[history, tensorboard],
                validation_data=read_hdf5(
                        hdf5_file,
                        dataset="test",
                        batch_size=batch_size,
                        ),
                validation_steps=validation_steps
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

    except StopIteration:
        hdf5_file.close()
    finally:
        hdf5_file.close()

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
    logging.basicConfig(level=logging.DEBUG)
    main()
