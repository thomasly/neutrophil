# -*- coding: utf-8 -*-

import os, h5py
from random import shuffle
import matplotlib.pyplot as plt
from math import ceil

def read_hdf5(hdf5_file, dataset="train", batch_size=32, epochs=1):
    """
    """
    
    data_img = dataset + "_img"
    data_labels = dataset + "_labels"
    
    m_data = hdf5_file[data_img].shape[0]
    batch_list = list(range(int(ceil(m_data / batch_size))))
    
    while True:
        shuffle(batch_list)
        for idx, num in enumerate(batch_list):
            n_start = num * batch_size
            n_end = min((num + 1) * batch_size, m_data)
            
            inputs = hdf5_file[data_img][n_start:n_end, ...]
            targets = hdf5_file[data_labels][n_start:n_end]
            yield inputs, targets
        


if __name__ == "__main__":
    path = os.path.join("..", "data", "76_79_80.hdf5")
    hdf5_file = h5py.File(path, mode='r')
    for image, label in read_hdf5(path):
        if len(label) < 32:
            print("Label: ", str(label))
            plt.imshow(image[0])
            plt.show()
            
    for image, label in read_hdf5(path, "test"):
        if len(label) < 32:
            print("Label: ", str(label))
            plt.imshow(image[0])
            plt.show()
    hdf5_file.close()