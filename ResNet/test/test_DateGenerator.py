# import importlib.util, os
# spec = importlib.util.spec_from_file_location("DataGenerator", "../DataGenerator.py")
# DataGenerator = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(DataGenerator)
import sys
sys.path.append("..")
from DataGenerator import DataGenerator
import os

def test_data_generator():

    hdf5_file = os.path.join("..", "..", "data", "76_79_80.hdf5")
    params = {
        "batch_size" : 32,
        "shuffle" : True,
        "n_classes" : 2
    }
    generator = DataGenerator(hdf5_file, "train", **params)
    print("Batch size: ", generator.batch_size)
    print("Data dim: ", generator.dim)
    print("Classes: ", generator.n_classes)
    print("Data number: ", generator.n_data)
    print("list_IDs length: ", len(generator.list_IDs))
    print("Training steps: ", len(generator))
    steps = len(generator)
    # mini batch not at the end of the dataset
    dataset, labels = generator.__getitem__(steps - 2)
    print("Data 1 shape: ", dataset[0].shape)
    print("Data 32 shape: ", dataset[31].shape)
    print("Label 1: ", labels[0])
    print("Label 32: ", labels[31])
    try:
        dataset[32]
    except IndexError:
        print("Dataset size is correct")
        print()
        # raise
    # the last mini batch, size usually is smaller than batch_size
    dataset, labels = generator.__getitem__(steps - 1)
    print("Batch size is:", params["batch_size"])
    print("The size of the last mini batch is:", len(dataset))
    l = len(dataset)
    print("Data 1 shape:", dataset[0].shape)
    print("Data {} (last data) shape:".format(l), dataset[l-1].shape)
    print("Label 1:", labels[0])
    print("Label {}:".format(l), labels[l-1])
    try:
        dataset[l]
    except IndexError:
        print("Dataset size is correct")
        # raise

if __name__ == '__main__':
    test_data_generator()