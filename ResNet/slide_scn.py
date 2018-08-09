# -*- coding: utf-8 -*-

from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import os, time
import tables as tb
import numpy as np
from datetime import datetime
import multiprocessing as mp

def v_slide(params):
    """
    """
    
    try:
        home_path = os.path.abspath("..")
        file_path = os.path.join(
                home_path, 
                "data", 
                "orig", 
                "ImageCollection_80.scn")

        try:
            scn_file = OpenSlide(file_path)
            
        except OpenSlideUnsupportedFormatError:
            print("OpenSlideUnsupportedFormatError!")
            return
        except OpenSlideError:
            print("OpenSlideError!")
            return
        
        start_point = params["start_point"]
        x0 = start_point[0]
        y0 = start_point[1]
        bound_y =  params["bound_y"]
        tile_path = params["tile_path"]
        save_tiles = params["save_tiles"]
        q = params["queue"]
        
        STD_THRESHOLD = 40
        pid = os.getpid()
        data = {}
        while y0 < bound_y:
            img = scn_file.read_region((x0, y0), 0, (299, 299))
            std = np.std(img)
            if std > STD_THRESHOLD:
                sufix = "_" + str(x0) + "_" + \
                        str(y0) + ".png"
                file_name = "scn80" + sufix
                img = np.array(img)
                img = img[:, :, 0:3]
                data['pred'] = img
                data['xlabel'] = np.array([x0])
                data['ylabel'] = np.array([y0])
                q.put(dict(data))
                if save_tiles:
                    img.save(os.path.join(tile_path, file_name))
                    
            y0 += 150
        
        return pid
    finally:
        scn_file.close()
    
def listener(q):
    """
    """
    try:
        counter = 0
        pid = os.getpid()
        print("Listener running on {}".format(pid))
        home_path = os.path.abspath('..')
        hdf5_path = os.path.join(home_path, "data", "test", "tiles_80.hdf5")
        hdf5_file = tb.open_file(hdf5_path, mode='w')
        pred_storage = hdf5_file.create_earray(hdf5_file.root, 
                                               "pred_img", 
                                               tb.UInt8Atom(), 
                                               shape=(0, 299, 299, 3))
        xlabel_storage = hdf5_file.create_earray(hdf5_file.root, 
                                                 "pos_xlabel", 
                                                 tb.UInt16Atom(), 
                                                 shape=(0,1))
        ylabel_storage = hdf5_file.create_earray(hdf5_file.root, 
                                                 "pos_ylabel", 
                                                 tb.UInt16Atom(), 
                                                 shape=(0,1))
        
        while 1:
            counter += 1
            if counter % 100 == 0:
                print("{} tiles saved in hdf5.".format(counter), end = "\r")
            data = q.get()
            if data == 'kill':
                print("Listner closed.")
                break
            pred = data['pred']
            xlabel = data['xlabel']
            ylabel = data['ylabel']
            
            pred_storage.append(pred[None])
            xlabel_storage.append(xlabel[None])
            ylabel_storage.append(ylabel[None])
    finally:
        hdf5_file.close()
    
def slide_scn(scn_file=None, save_tiles=False):
    """
    Slide the whole scn file into tiles. Tiles sizes are (299, 299). 
    Tiles have a half of the tile width overlapping with the tiles beside them
    to ensure all regions are well covered without neutrophil locating on the 
    side of the tiles to be ignored in the later prediction.
    The function use multiprocessing method to increase sliding speed.
    Tiles are saved into hdf5 file after sliding.
    
    input:
        scn_file - path to scn file
        save_tiles - save tiles images or not
        
    output:
        hdf5 file saved in neutrophil/data/test folder.
        Label name is "pred_img".
        Tiles will be saved in the save folder if save_tiles flag is set to True.
    """
    
    # to get the running time
    start_time = datetime.now()
    
    # default scn file path, should be replaced with sys.argv 
    # when the project is done.
    home_path = os.path.abspath("..")
    file_path = os.path.join(
            home_path, 
            "data", 
            "orig", 
            "ImageCollection_80.scn")
    
    # path to the folder saving tiles and hdf5 file.
    tile_path = os.path.join(home_path, "data", "test")
    
    # open scn_file
    if not scn_file:
        try:
            scn_file = OpenSlide(file_path)
            
        except OpenSlideUnsupportedFormatError:
            print("OpenSlideUnsupportedFormatError!")
            return
        except OpenSlideError:
            print("OpenSlideError!")
            return
    else:
        try:
            scn_file = OpenSlide(scn_file)
            
        except OpenSlideUnsupportedFormatError:
            print("OpenSlideUnsupportedFormatError!")
            return
        except OpenSlideError:
            print("OpenSlideError!")
    
    # get attributes of the scn_file
    x0 = int(scn_file.properties["openslide.bounds-x"])
    y0 = int(scn_file.properties["openslide.bounds-y"])
    width = int(scn_file.properties["openslide.bounds-width"])
    height = int(scn_file.properties["openslide.bounds-height"])
    bound_x = x0 + width - 150
    bound_y = y0 + height - 150
    start_point = [x0 + 150, y0 + 150]
    scn_file.close()
    
    # create multiporcessing pool
    pool = mp.Pool(mp.cpu_count())
    manager = mp.Manager()
    q = manager.Queue()
    watcher = pool.apply_async(listener, (q, ))
    
    # parameters passed to pool.map() function need to be packed in a list
    tasks = []
    task = {
            "bound_y": bound_y,
            "tile_path": tile_path,
            "save_tiles": save_tiles,
            "queue": q
            }
    while start_point[0] < bound_x:
        task["start_point"] = list([start_point[0], start_point[1]])
        tasks.append(dict(task))
        start_point[0] += 150
    
    # slide images with multiprocessing
    jobs = []
    start_watch = True
    for task in tasks:
        job = pool.apply_async(v_slide, (task, ))
        jobs.append(job)
        
    for job in jobs:
        job.get()
        if start_watch:
            start_watch = False
            watcher.get()
    
    q.put('kill')
    pool.close()
    print("Done!")
    print("Time consumed: {}".format(datetime.now() - start_time))
    
#    h5_file_path = os.path.join(tile_path, "pred_img.hdf5")
#    hdf5_file = h5py.File(h5_file_path, mode = 'w')
#    files = glob.glob(tile_path + os.sep + "*.png")
#    n_files = len(files)
#    
#    shape = (n_files, 299, 299, 3)
#    
#    hdf5_file.create_dataset("pred_img", shape, np.int8)
#    
#    for i in range(n_files):
#        if i % 500 == 0 and i > 1:
#            print("Tiles: {}/{}".format(i, n_files))
#            
#        x = files[i]
#        img = Image.open(x)
#        img = np.array(img)
#        img = img[:, :, 0:3]
#        hdf5_file["pred_img"][i, ...] = img
#        
#    hdf5_file.close()
    
    
    
    
            
if __name__ == "__main__":
    slide_scn(save_tiles=False)