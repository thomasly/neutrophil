# -*- coding: utf-8 -*-

from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import os, h5py, glob
import numpy as np
from datetime import datetime
from PIL import Image
import multiprocessing as mp

def v_slide(params):
    """
    """
    
    start_point = params["start_point"]
    bound_y =  params["bound_y"]
    scn_file = params["scn_file"]
    tile_path = params["tile_path"]
    
    STD_THRESHOLD = 40
    while start_point[1] < bound_y:
        img = scn_file.read_region(start_point, 0, (299, 299))
        std = np.std(img)
        if std < STD_THRESHOLD:
            print("Center position: ({}, {}). \
Background region!".format(
                  start_point[0], 
                  start_point[1]))
        else:
            sufix = "_" + str(start_point[0]) + "_" + \
                    str(start_point[1]) + ".png"
            file_name = "scn80" + sufix
            img.save(os.path.join(tile_path, file_name))  
        start_point[1] += 150
    

def slide_scn(scn_file=None):
    """
    """
    
    start_time = datetime.now()
    home_path = os.path.abspath("..")
    file_path = os.path.join(
            home_path, 
            "data", 
            "orig", 
            "ImageCollection_80.scn")
    tile_path = os.path.join(home_path, "data", "test")
    if not scn_file:
        try:
            scn_file = OpenSlide(file_path)
            
        except OpenSlideUnsupportedFormatError:
            print("OpenSlideUnsupportedFormatError!")
            return
        except OpenSlideError:
            print("OpenSlideError!")
            return
        
    x0 = int(scn_file.properties["openslide.bounds-x"])
    y0 = int(scn_file.properties["openslide.bounds-y"])
    width = int(scn_file.properties["openslide.bounds-width"])
    height = int(scn_file.properties["openslide.bounds-height"])
    bound_x = x0 + width - 150
    bound_y = y0 + height - 150
    start_point = [x0 + 150, y0 + 150]
    
    pool = mp.Pool(mp.cpu_count())
    tasks = []
    task = {
            "bound_y": bound_y,
            "scn_file": scn_file,
            "tile_path": tile_path
            }
    while start_point[0] < bound_x:
        task["start_point"] = list([start_point[0], start_point[1]])
        tasks.append(dict(task))
        start_point[0] += 150
        
    pool.map(v_slide, tasks)
        
    pool.close()
    pool.join()
    
    scn_file.close()
    print("Done!")
    print("Time consumed: {}".format(datetime.now() - start_time))
    
    h5_file_path = os.path.join(tile_path, "pred_img.hdf5")
    hdf5_file = h5py.File(h5_file_path, mode = 'w')
    files = glob.glob(tile_path + os.sep + "*.png")
    n_files = len(files)
    
    shape = (n_files, 299, 299, 3)
    
    hdf5_file.create_dataset("pred_img", shape, np.int8)
    
    for i in range(n_files):
        if i % 500 == 0 and i > 1:
            print("Tiles: {}/{}".format(i, n_files))
            
        x = files[i]
        img = Image.open(x)
        img = np.array(img)
        img = img[:, :, 0:3]
        hdf5_file["pred_img"][i, ...] = img
        
    hdf5_file.close()
    
    
    
    
            
if __name__ == "__main__":
    slide_scn()