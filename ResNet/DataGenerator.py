import keras
import numpy as np
import tables as tb

class DataGenerator(keras.utils.Sequence):

    def __init__(
        self,
        hdf5_file_name,
        dataset,
        batch_size=32,
        n_classes=2,
        shuffle=True
    ):

        """
        Initialization
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.hdf5_file_name = hdf5_file_name
        self.data_img = dataset + "_img"
        self.data_labels = dataset + "_labels"
        with tb.open_file(self.hdf5_file_name, "r") as hdf5_file:
            data_shape = hdf5_file.root.__getitem__(self.data_img).shape
            self.n_data = len(data_shape[0])
            self.dim = data_shape[1:3]
            self.n_channels = data_shape[3]
            self.list_IDs = np.arange(self.n_data)
        
    
    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        # generate data
        with tb.open_file(self.hdf5_file_name, 'r') as hdf5_file:
            for i, ID in enumerate(list_IDs_temp):
                # store samples
                X[i,] = hdf5_file.root.__getitem__(self.data_img)[ID]

                # store class
                Y[i] = hdf5_file.root.__getitem__(self.data_labels)[ID]

        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)

    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self.n_data / self.batch_size))


    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        batch_indices = self.indices[
            index * self.batch_size : min(
                (index + 1) * self.batch_size, self.n_data)
        ]

        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in batch_indices]

        # generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y


        