# -*- coding: utf-8 -*-

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import os, tables, sys
from math import ceil
from LossHistory import LossHistory
from DataGenerator import DataGenerator
from datetime import datetime, date
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model
import tensorflow as tf
import argparse
from paths import Paths


def generate_model(
    model_name, 
    input_shape, 
    include_top=False, 
    weights=None, 
    classes=1000,
    pooling=None):
    """
    return keras model
    """

    models = {
        "InceptionResNetV2" : InceptionResNetV2,
        "ResNet50" : ResNet50,
        "InceptionV3" : InceptionV3
    }
    if include_top:
        if weights:
            model = models[model_name](
                weights=weights, 
                include_top=include_top
            )
        else:
            model = models[model_name](
                weights=weights, 
                include_top=include_top,
                classes=classes
            )
    
    elif not include_top:
        model = models[model_name](
            weights=weights, 
            include_top=include_top, 
            classes=classes,
            input_shape=input_shape,
            pooling=pooling
        )
        # add top
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation="relu"))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(classes, activation="softmax"))
        model.add(top_model)
        
    return model


def train(
    model_name, 
    batch_size = 32, 
    epochs = 10, 
    classes=2, 
    n_gpu = 8, 
    validation = True):
    """
    """
    
    start_time = datetime.now()
    paths = Paths()
    hdf5_path = os.path.join("..", "data", "76_79_80.hdf5")
    if n_gpu > 1:
        # create model on cpu
        print("Using {} gpus...".format(n_gpu))
        with tf.device("/cpu:0"):
            model = generate_model(
                model_name, 
                input_shape=(299, 299, 3), 
                include_top=False, 
                weights=None, 
                classes=classes, 
                pooling=None
            )
            print("Generated model on cpu...")
            sys.stdout.flush()

        # generate mutiple models to use multi gpus
        model = multi_gpu_model(model, gpus=n_gpu)
        print("Created parallel model...")
        sys.stdout.flush()

    else:
        print("Using single gpu...")
        model = generate_model(
            model_name, 
            input_shape=(299, 299, 3), 
            include_top=False, 
            weights=None, 
            classes=2, 
            pooling=None
        )
        print("Generated model on cpu...")
        sys.stdout.flush()

    model.compile(
            optimizer = "adam", 
            loss = "categorical_crossentropy", 
            metrics = ["accuracy"]
            )
    print("Model compiled...")
    sys.stdout.flush()

    timestamp = datetime.now().strftime(r"%Y%m%d_%I%M%p")
    tb_log_path = os.path.join(paths.logs, 'logs_{}'.format(timestamp))
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
    
    tensorboard = TensorBoard(log_dir = tb_log_path)
    print("Start training...")
    sys.stdout.flush()

    data_params = {
        "batch_size" : batch_size,
        "n_classes" : classes,
        "shuffle" : True
    }
    train_generator = DataGenerator(hdf5_path, "train", **data_params)
    valid_generator = DataGenerator(hdf5_path, "test", **data_params)
    model.fit_generator(
            train_generator,
            epochs = epochs,
            verbose = 2, 
            callbacks = [history, tensorboard],
            validation_data = valid_generator
            # use_multiprocessing = False
            # workers = 1
            # max_queue_size = 5
            )
    
    if validation:
        pred_generator = DataGenerator(hdf5_path, "val", **data_params)
        preds = model.evaluate_generator(
                pred_generator
                # use_multiprocessing = False,
                # workers = 8
                )
        print("Validation loss: {}".format(preds[0]))
        print("Validation accuracy: {}".format(preds[1]))
    
    time_consumed = datetime.now() - start_time
    hours = time_consumed // 3600
    minutes = time_consumed % 3600 // 60
    seconds = time_consumed % 60 
    print("Training time: {}h{}m{}s".format(hours, minutes, seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument(
        "-n",
        "--model-name",
        choices=["ResNet50", "InceptionV3", "InceptionResNetV2"],
        help="Name of the model."
    )
    parser.add_argument(
        "-b",
        "--batch-size", 
        type=int,
        default=32,
        help="Size of mini batches."
    )
    parser.add_argument(
        "-e", 
        "--epochs", 
        type=int, 
        default=20, 
        help="Number of epochs."
    )
    parser.add_argument(
        "-g", 
        "--n-gpu", 
        type=int, 
        default=1, 
        help="Number of gpus."
    )
    parser.add_argument(
        "-v",
        "--validation",
        action="store_true"
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        default=2,
        help="Number of classes."
    )
    args = vars(parser.parse_args())

    print(args)
    train(**args)