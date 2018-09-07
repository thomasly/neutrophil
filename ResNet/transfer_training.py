# -*- coding: utf-8 -*-

#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
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


def train(batch_size = 32, epochs = 10, n_gpu = 8, validation = True):
    """
    """
    
    start_time = datetime.now()
    hdf5_path = os.path.join("..", "data", "76_79_80.hdf5")
    with tf.device("/cpu:0"):
        model = InceptionV3(weights='imagenet', include_top=False)
        inputs = Input(shape=(299, 299, 3))
        model_output = model(inputs)
        X = Flatten(name="flatten")(model_output)
        X = Dense(128, activation='relu', name="dense")(X)
        output = Dense(2, activation='softmax', name="classifier")(X)
        model = Model(inputs=inputs, outputs=output)
        print("Generated model on cpu...")
    parallel_model = multi_gpu_model(model, gpus=n_gpu)
    print("Created parallel model...")
    sys.stdout.flush()

    parallel_model.compile(
            optimizer = "adam", 
            loss = "binary_crossentropy", 
            metrics = ["accuracy"]
            )
    print("Model compiled...")
    sys.stdout.flush()

    timestamp = datetime.now().strftime(r"%Y%m%d%I%M")
    history = LossHistory('epoch_loss_{}.log'.format(timestamp), 
                    'batch_loss_{}.log'.format(timestamp),
                    'inceptionModel_{}.json'.format(timestamp),
                    'inceptionWeights_{}.h5'.format(timestamp))
    try:
        os.mkdir('./logs_{}'.format(timestamp))
    except IOError:
        pass
    tensorboard = TensorBoard(log_dir="./logs_{}".format(timestamp))
    print("Start training...")
    sys.stdout.flush()

    data_params = {
        "batch_size" : batch_size,
        "n_classes" : 2,
        "shuffle" : True
    }
    train_generator = DataGenerator(hdf5_path, "train", **data_params)
    valid_generator = DataGenerator(hdf5_path, "test", **data_params)
    parallel_model.fit_generator(
            train_generator,
            epochs = epochs,
            verbose = 2, 
            callbacks = [history, tensorboard],
            validation_data = valid_generator,
            use_multiprocessing = True,
            workers = 2
            # max_queue_size = 5
            )
    
    if validation:
        pred_generator = DataGenerator(hdf5_path, "val", **data_params)
        preds = parallel_model.evaluate_generator(
                pred_generator,
                use_multiprocessing = True,
                workers = 8
                )
        print("Validation loss: {}".format(preds[0]))
        print("Validation accuracy: {}".format(preds[1]))
        
    print("Training time: ", datetime.now() - start_time)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network.")
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
        default=8, 
        help="Number of gpu."
    )
    parser.add_argument(
        "-v",
        "--validation",
        action="store_true"
    )
    args = vars(parser.parse_args())

    print(args)
    train(**args)