# -*- coding: utf-8 -*-

import os, h5py
from math import ceil
from load_model_from_file import load_model_from_json
import numpy as np

def read_hdf5(hdf5_file, dataset="train", batch_size=32, epochs=1):
    """
    """
    
    data_img = dataset + "_img"
    
    m_data = hdf5_file[data_img].shape[0]
    batch_list = list(range(int(ceil(m_data / batch_size))))
    
    while True:
        for num in batch_list:
            n_start = num * batch_size
            n_end = min((num + 1) * batch_size, m_data)
            inputs = hdf5_file[data_img][n_start:n_end, ...]
            yield inputs
            
def pred(hdf5_file_path=None):
    """
    """
    try:
        BATCH_SIZE = 32
        model = load_model_from_json()
        home = os.path.abspath("..")
        default_path = os.path.join(home, 'data', 'test', 'tiles_80.hdf5')
        if hdf5_file_path:
            hdf5_file = h5py.File(hdf5_file_path, mode='r')
            
        else:
            hdf5_file = h5py.File(default_path, mode='r')
        
        m_samples = hdf5_file["pred_img"].shape[0]
        steps = int(ceil(m_samples / BATCH_SIZE))
        generator = read_hdf5(hdf5_file, dataset="pred", batch_size=BATCH_SIZE)
        preds = model.predict_generator(generator, steps=steps, verbose=1)
        
        hdf5_file.close()
        print(preds[0:100])
        
    finally:
        
        csv_path = os.path.join(home, 'data', 'test', 'preds.csv')
        np.savetxt(csv_path, preds, delimiter=',')
        hdf5_file.close()
    

if __name__ == "__main__":
    pred()