# -*- coding: utf-8 -*-

import os, sys
from math import ceil
from load_model_from_file import load_model_from_json
import numpy as np
import tables as tb
from paths import Paths

def read_hdf5(hdf5_file, dataset="train", batch_size=32):
    """
    """
    
    data_img = dataset + "_img"
    
    m_data = hdf5_file.root.__getitem__(data_img).shape[0]
    batch_list = list(range(int(ceil(m_data / batch_size))))
    
    while True:
        for num in batch_list:
            n_start = num * batch_size
            n_end = min((num + 1) * batch_size, m_data)
            inputs = hdf5_file.root.__getitem__(data_img)[n_start:n_end, ...]
            yield inputs
            
def pred(model_path, hdf5_file_path=None):
    """
    """
    try:
        paths = Paths()
        BATCH_SIZE = 32
        model = load_model_from_json()
        default_path = paths.tiles_80
        if hdf5_file_path:
            hdf5_file = tb.open_file(hdf5_file_path, mode='r')
            
        else:
            hdf5_file = tb.open_file(default_path, mode='r')
        
        m_samples = hdf5_file.root.__getitem__("pred_img").shape[0]
        steps = int(ceil(m_samples / BATCH_SIZE))
        generator = read_hdf5(hdf5_file, dataset="pred", batch_size=BATCH_SIZE)
        preds = model.predict_generator(generator, steps=steps, verbose=1)
        
        hdf5_file.close()
        print(preds[0:100])
        
    finally:
        csv_path = paths.pred_cvs
        np.savetxt(csv_path, preds, delimiter=',')
        hdf5_file.close()
    

if __name__ == "__main__":
    pred("./models/{}".format(sys.argv[1]))