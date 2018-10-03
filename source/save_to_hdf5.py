'''
Filename: save_to_hdf5.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 3, 2018 3:57 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import h5py
import os
import glob
from PIL import Image
import numpy as np
from random import shuffle
import tables as tb


def save_to_hdf5(path):
    """
    """
    TRAINING_DATA_PECENTAGE = 0.8
    TESTING_DATA_PECENTAGE = 0.1

    src_path = os.path.abspath(path)
    des_path = os.path.join(src_path, "..")
    des_path = os.path.join(des_path, "76_79_80.hdf5")

    files_path = os.path.join(src_path, "*.png")
    files = glob.glob(files_path)
    labels = [1 if "pos" in file_name else 0 for file_name in files]

    c = list(zip(files, labels))
    shuffle(c)
    files, labels = zip(*c)

    m = len(files)
    part_train = int(m * TRAINING_DATA_PECENTAGE)
    part_test = part_train + int(m * TESTING_DATA_PECENTAGE)
    x_train = files[0: part_train]
    x_test = files[part_train: part_test]
    x_val = files[part_test:]
    y_train = labels[0: part_train]
    y_test = labels[part_train: part_test]
    y_val = labels[part_test:]

    train_shape = (len(x_train), 299, 299, 3)
    test_shape = (len(x_test), 299, 299, 3)
    val_shape = (len(x_val), 299, 299, 3)

    hdf5_file = h5py.File(des_path, mode='w')
    m_train = len(x_train)
    m_test = len(x_test)
    m_val = len(x_val)

    hdf5_file.create_dataset("train_img", train_shape, np.uint8)
    hdf5_file.create_dataset("test_img", test_shape, np.uint8)
    hdf5_file.create_dataset("val_img", val_shape, np.uint8)

    hdf5_file.create_dataset("train_labels", (m_train,), np.uint8)
    hdf5_file.create_dataset("test_labels", (m_test,), np.uint8)
    hdf5_file.create_dataset("val_labels", (m_val,), np.uint8)

    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    hdf5_file["train_labels"][...] = y_train
    hdf5_file["test_labels"][...] = y_test
    hdf5_file["val_labels"][...] = y_val

    mean = np.zeros(train_shape[1:], np.float32)

    for i in range(m_train):
        if i % 500 == 0 and i > 1:
            print("Train data: {}/{}".format(i, m_train))

        x = x_train[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        hdf5_file["train_img"][i, ...] = img
        mean += img / float(m_train)

    for i in range(m_test):
        if i % 100 == 0 and i > 1:
            print("Test data: {}/{}".format(i, m_test))

        x = x_test[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        hdf5_file["test_img"][i, ...] = img

    for i in range(m_val):
        if i % 100 == 0 and i > 1:
            print("Validation data: {}/{}".format(i, m_val))

        x = x_val[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        hdf5_file["val_img"][i, ...] = img

    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()


def save_to_hdf5_with_tables(path):
    """
    """
    TRAINING_DATA_PECENTAGE = 0.8
    TESTING_DATA_PECENTAGE = 0.1

    src_path = os.path.abspath(path)
    des_path = os.path.join(src_path, "..")
    des_path = os.path.join(des_path, "76_79_80.hdf5")

    files_path = os.path.join(src_path, "*.png")
    files = glob.glob(files_path)
    labels = [1 if "pos" in file_name else 0 for file_name in files]

    c = list(zip(files, labels))
    shuffle(c)
    files, labels = zip(*c)

    m = len(files)
    part_train = int(m * TRAINING_DATA_PECENTAGE)
    part_test = part_train + int(m * TESTING_DATA_PECENTAGE)
    x_train = files[0: part_train]
    x_test = files[part_train: part_test]
    x_val = files[part_test:]
    y_train = labels[0: part_train]
    y_test = labels[part_train: part_test]
    y_val = labels[part_test:]

    data_shape = (0, 299, 299, 3)

    hdf5_file = tb.open_file(des_path, mode='w')
    m_train = len(x_train)
    m_test = len(x_test)
    m_val = len(x_val)

    train_storage = hdf5_file.create_earray(
        hdf5_file.root, "train_img", tb.UInt8Atom(), shape=data_shape
    )
    test_storage = hdf5_file.create_earray(
        hdf5_file.root, "test_img", tb.UInt8Atom(), shape=data_shape
    )
    val_storage = hdf5_file.create_earray(
        hdf5_file.root, "val_img", tb.UInt8Atom(), shape=data_shape
    )

    hdf5_file.create_array(hdf5_file.root, "train_labels", y_train)
    hdf5_file.create_array(hdf5_file.root, "test_labels", y_test)
    hdf5_file.create_array(hdf5_file.root, "val_labels", y_val)

    mean_storage = hdf5_file.create_earray(
        hdf5_file.root, "train_mean", tb.Float32Atom(), shape=data_shape
    )

    mean = np.zeros(data_shape[1:], np.float32)

    for i in range(m_train):
        if i % 500 == 0 and i > 1:
            print("Train data: {}/{}".format(i, m_train))

        x = x_train[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        train_storage.append(img[None])
        mean += img / float(m_train)

    for i in range(m_test):
        if i % 100 == 0 and i > 1:
            print("Test data: {}/{}".format(i, m_test))

        x = x_test[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        test_storage.append(img[None])

    for i in range(m_val):
        if i % 100 == 0 and i > 1:
            print("Validation data: {}/{}".format(i, m_val))

        x = x_val[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        val_storage.append(img[None])

    mean_storage.append(mean[None])
    hdf5_file.close()


if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()
    home = os.path.abspath("..")
    pool_path = os.path.join(home, "data", "pool")
    save_to_hdf5_with_tables(pool_path)
    end_time = timeit.default_timer()

    print("Time consumed: {}".format(end_time - start_time))
