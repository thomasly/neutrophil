# -*- coding: utf-8 -*-

import os
import tables as tb
import pandas as pd
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

def pool_img_based_on_pred(target_hdf5_file = None, target_pred_csv_file = None):
    """
    """
    
    home = os.path.abspath("..")
    save_path = os.path.join(home, 'data', 'predict')
    try:
        os.mkdir(save_path)
    except OSError:
        pass
    
    if not target_hdf5_file:
        hdf5_file_path = os.path.join(home, 'data', 'test', 'tiles_80.hdf5')
    else:
        hdf5_file_path = os.path.join(home, 'data', target_hdf5_file)
        
    if not target_pred_csv_file:
        csv_file_path = os.path.join(home, 'data', 'test', 'preds.csv')
    else:
        csv_file_path = os.path.join(home, 'data', target_pred_csv_file)
        
    sufix = os.path.split(hdf5_file_path)[1].split('.')[0]
    try:    
        hdf5_file = tb.open_file(hdf5_file_path, mode = 'r')
        pred_img = hdf5_file.root.pred_img
        x_labels = hdf5_file.root.pos_xlabel
        y_labels = hdf5_file.root.pos_ylabel
        
        pred_df = pd.read_csv(csv_file_path, sep=',', header=None, names=['preds']) 
        
        for thresh in np.arange(0.1, 1.15, 0.1):
            try:
                dir_path = os.path.join(save_path, "%.1f"%thresh + '_' + "%.1f"%(thresh + 0.1))
                os.mkdir(dir_path)
            except OSError:
                pass
            
            ind = pred_df.loc[pred_df['preds'] >= thresh]
            ind = ind.index[ind['preds'] < (thresh + 0.1)].tolist()
            counter = 0
            for i in ind:
                counter += 1
                img = Image.fromarray(pred_img[i].astype('uint8'))
                if counter % 100 == 0:
                    print('{} images saved in {}'.format(counter, dir_path))
                img_name = str(x_labels[i]) + '_' + str(y_labels[i]) + sufix + '.png'
                img.save(os.path.join(dir_path, img_name))
                
    finally:
        hdf5_file.close()
        
if __name__ == '__main__':
    pool_img_based_on_pred()