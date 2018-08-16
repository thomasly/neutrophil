# -*- coding: utf-8 -*-

import os, h5py
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

def read_hdf5(hdf5_file, dataset="train", batch_size=32):
    """
    """
    
    data_img = dataset + "_img"
    data_labels = dataset + "_labels"
    
    m_data = hdf5_file.root.__getitem__(data_img).shape[0]
    ix = list(range(m_data))
    
    while True:
        shuffle(ix)
        inputs = []
        targets = []
        for i in range(m_data):
            inputs.append(hdf5_file.root.__getitem__(data_img)[ix[i]])
            targets.append(to_categorical(hdf5_file.root.__getitem__(data_labels)[ix[i]], num_classes=2))
            if (i+1) % batch_size == 0 or (i+1) == m_data:
                inputs = np.stack(inputs)
                targets = np.stack(targets)
                yield inputs, targets
                inputs = []
                targets = []
        


if __name__ == "__main__":
    path = os.path.join("..", "data", "76_79_80.hdf5")
    hdf5_file = h5py.File(path, mode='r')
    for image, label in read_hdf5(hdf5_file):
        plt.imshow(image[0])
        break
    hdf5_file.close()