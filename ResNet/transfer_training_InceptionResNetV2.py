# -*- coding: utf-8 -*-

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import os, tables
from math import ceil
from LossHistory import LossHistory
from read_hdf5 import read_hdf5
from tensorflow.keras.utils import to_categorical
from datetime import datetime

def train(batch_size = 32, epochs = 15, validation = True):
    """
    """
    
    start_time = datetime.now()
    hdf5_path = os.path.join("..", "data", "76_79_80.hdf5")
    model = InceptionResNetV2(classes = 1000)
    model.compile(
            optimizer = "adam", 
            loss = "binary_crossentropy", 
            metrics = ["accuracy"]
            )
    
    hdf5_file = tables.open_file(hdf5_path, mode = 'r')
    n_train = hdf5_file.root.train_img.shape[0]
    n_test = hdf5_file.root.test_img.shape[0]
    
    steps_per_epoch = int(ceil(n_train / batch_size))
    validation_steps = int(ceil(n_test / batch_size))

    history = LossHistory('epoch_loss.log', 'batch_loss.log')
    
    try:
        model.fit_generator(
                read_hdf5(
                        hdf5_file, 
                        batch_size = batch_size,
                        ), 
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                verbose = 2, 
                callbacks = [history],
                validation_data = read_hdf5(
                        hdf5_file, 
                        dataset = "test", 
                        batch_size = batch_size,
                        ),
                validation_steps = validation_steps
                )
        
        if validation:
            eva_data = hdf5_file.root.val_img
            eva_labels = hdf5_file.root.val_labels
            eva_labels = to_categorical(eva_labels, num_classes=1000)
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
    
if __name__ == '__main__':
    train()