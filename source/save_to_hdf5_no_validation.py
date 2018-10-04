'''
Filename: save_to_hdf5_no_validation.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Oct 3, 2018 3:58 PM
-----
Last Modified: Oct 3, 2018 5:40 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import os
import logging
from PIL import Image
import numpy as np
import tables as tb
from random import shuffle


def save_to_hdf5_with_tables(path):
    """
    """
    TRAINING_DATA_PECENTAGE = 0.9

    src_path = os.path.abspath(path)
    des_path = os.path.join(src_path, os.pardir)
    des_path = os.path.join(des_path, "76_79_80_noValidate.hdf5")

    files = os.scandir(src_path)
    grouped_files = {}
    for f in files:
        if f.name.endswith('.png'):
            region = f.name.split('_')[2] + f.name.split('_')[3]
            logging.debug(f'Region: {region}')
        if region not in grouped_files.keys():
            grouped_files[region] = []
            grouped_files[region].append(f.path)
        else:
            grouped_files[region].append(f.path)

    n_groups = len(grouped_files)
    logging.info(f'n_groups is: {n_groups}')
    grouped_values = list(grouped_files.values())
    shuffle(grouped_values)
    part_train = int(n_groups * TRAINING_DATA_PECENTAGE)
    logging.info(f'part_train is: {part_train}')

    data_shape = (0, 299, 299, 3)

    hdf5_file = tb.open_file(des_path, mode='w')

    train_storage = hdf5_file.create_earray(
        hdf5_file.root, "train_img", tb.UInt8Atom(), shape=data_shape
    )
    test_storage = hdf5_file.create_earray(
        hdf5_file.root, "test_img", tb.UInt8Atom(), shape=data_shape
    )

    train_label_storage = hdf5_file.create_earray(
        hdf5_file.root, "train_labels", tb.UInt8Atom(), shape=(0,)
    )
    test_label_storage = hdf5_file.create_earray(
        hdf5_file.root, "test_labels", tb.UInt8Atom(), shape=(0,)
    )

    mean_storage = hdf5_file.create_earray(
        hdf5_file.root, "train_mean", tb.Float32Atom(), shape=data_shape
    )

    mean = np.zeros(data_shape[1:], np.float32)

    for i in range(part_train):
        if i % 125 == 0 and i > 1:
            logging.warning("Train data: {}/{}".format(i * 4, part_train * 4))
        group = grouped_values[i]
        for x in group:
            img = Image.open(x)
            img = np.array(img)
            img = img[:, :, 0:3]
            train_storage.append(img[None])

            if 'pos' in x:
                label = 1
                if i % 5 == 0:
                    logging.info(f'Label in train group is {label}')
            else:
                label = 0
                if i % 5 == 0:
                    logging.info(f'Label in train group is {label}')
            train_label_storage.append([label])

            mean += img / float(n_groups * 4)

    for i in range(part_train, n_groups):
        if i % 50 == 0 and i > 1:
            logging.warning(
                "Test data: {}/{}".format(
                    (i - part_train) * 4, (n_groups - part_train) * 4
                )
            )
        group = grouped_values[i]
        for x in group:
            img = Image.open(x)
            img = np.array(img)
            img = img[:, :, 0:3]
            test_storage.append(img[None])

            if 'pos' in x:
                label = 1
                if i % 5 == 0:
                    logging.info(f'Label in test group is {label}')
            else:
                label = 0
                if i % 5 == 0:
                    logging.info(f'Label in test group is {label}')
            test_label_storage.append([label])

    mean_storage.append(mean[None])
    hdf5_file.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    import timeit
    start_time = timeit.default_timer()
    file_path = os.path.abspath(__file__)
    logging.debug(f'File path {file_path}')
    home = os.path.join(os.path.dirname(file_path), os.pardir)
    pool_path = os.path.join(home, "data", "pool")
    logging.debug(f'pool path: {pool_path}')
    save_to_hdf5_with_tables(pool_path)
    end_time = timeit.default_timer()

    logging.warning("Time consumed: {}".format(end_time - start_time))
