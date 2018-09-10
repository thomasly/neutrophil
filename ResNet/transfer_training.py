# -*- coding: utf-8 -*-

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import os, tables, sys
from math import ceil
from LossHistory import LossHistory
from DataGenerator import DataGenerator
from datetime import datetime, date
from keras.layers import Input, Dense, Flatten
from keras.models import Model
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
        inputs = Input(shape=input_shape)
        model_output = model(inputs)
        X = Flatten(name="flatten")(model_output)
        X = Dense(128, activation="relu", name="dense_last")(X)
        outputs = Dense(classes, activation="softmax", name="classifier")(X)
        model = Model(inputs, outputs)
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
                pooling='max'
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
            pooling='max'
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
    history = LossHistory(
        '{}_epoch_loss_{}.log'.format(model_name, timestamp), 
        '{}_batch_loss_{}.log'.format(model_name, timestamp),
        '{}_{}.json'.format(model_name, timestamp),
        '{}_{}.h5'.format(model_name, timestamp)
    )             
    
    tb_log_path = os.path.join(paths.logs, 'logs_{}'.format(timestamp))
    try:
        os.makedirs(tb_log_path)
    except OSError:
        pass
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