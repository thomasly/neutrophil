'''
Filename: read_hdf5.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 3, 2018 5:55 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import os
import tables as tb
from random import shuffle
import numpy as np
from keras.utils import to_categorical
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def read_hdf5(hdf5_file, dataset="train", batch_size=32):
    """
    """

    data_img = dataset + "_img"
    data_labels = dataset + "_labels"

    m_data = hdf5_file.root.__getitem__(data_img).shape[0]
    indices = list(range(m_data))

    while True:
        shuffle(indices)
        inputs = []
        targets = []
        for i in range(m_data):
            inputs.append(hdf5_file.root.__getitem__(data_img)[indices[i]])
            targets.append(
                to_categorical(
                    hdf5_file.root.__getitem__(data_labels)[indices[i]],
                    num_classes=2
                )
            )
            if (i+1) % batch_size == 0 or (i+1) == m_data:
                inputs = np.stack(inputs)
                targets = np.stack(targets)
                yield inputs, targets
                inputs = []
                targets = []


if __name__ == "__main__":
    try:
        path = os.path.join("..", "data", "76_79_80_noValidate.hdf5")
        hdf5_file = tb.open_file(path, mode='r')
        for image, label in read_hdf5(hdf5_file):
            print(label[0])
            plt.imshow(image[0])
            plt.show()
    finally:
        hdf5_file.close()
