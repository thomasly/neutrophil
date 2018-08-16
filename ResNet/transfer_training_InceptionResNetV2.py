# -*- coding: utf-8 -*-

#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os, tables
from math import ceil
from LossHistory import LossHistory
from read_hdf5 import read_hdf5
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

def train(batch_size = 32, epochs = 10, validation = True):
    """
    """
    
    start_time = datetime.now()
    hdf5_path = os.path.join("..", "data", "76_79_80.hdf5")
    model = InceptionV3(weights='imagenet', include_top=False)
    
    inputs = Input(shape=(299, 299, 3))
    model_output = model(inputs)
    
    X = Flatten(name="flatten")(model_output)
    X = Dense(128, activation='relu', name="dense")(X)
    output = Dense(2, activation='softmax', name="classifier")(X)
    
    pretrained_model = Model(inputs=inputs, outputs=output)
    pretrained_model.compile(
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
        pretrained_model.fit_generator(
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
            n_val = hdf5_file.root.val_img.shape[0]
            val_steps = int(ceil(n_val / batch_size))
            preds = pretrained_model.evaluate_generator(
                    read_hdf5(
                            hdf5_file,
                            dataset="val",
                            batch_size=batch_size
                            ),
                    steps=val_steps
                    )
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