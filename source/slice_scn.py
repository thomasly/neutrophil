'''
Filename: slice_scn.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 10, 2018 5:43 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import os
import sys
import logging
from utils import timer
import tables as tb
import numpy as np
import multiprocessing as mp
from paths import Paths


def v_slide(params):
    """
    """

    paths = Paths()
    try:
        try:
            scn_file = OpenSlide(paths.slice_80)
        except OpenSlideUnsupportedFormatError:
            logging.error("OpenSlideUnsupportedFormatError!")
            return
        except OpenSlideError:
            logging.error("OpenSlideError!")
            return

        start_point = params["start_point"]
        x0 = start_point[0]
        y0 = start_point[1]
        bound_y = params["bound_y"]
        tile_path = params["tile_path"]
        save_tiles = params["save_tiles"]
        q = params["queue"]

        AVG_THRESHOLD = 170
        pid = os.getpid()
        data = {}
        while y0 < bound_y:
            img = scn_file.read_region((x0, y0), 0, (299, 299))
            green_c_avg = np.average(np.array(img)[:, :, 1])
            if green_c_avg < AVG_THRESHOLD:
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


def listener(q, output_path):
    """
    """
    try:
        counter = 0
        pid = os.getpid()
        print("Listener running on {}".format(pid))
        hdf5_file = tb.open_file(output_path, mode='w')
        pred_storage = hdf5_file.create_earray(
            hdf5_file.root,
            "pred_img",
            tb.UInt8Atom(),
            shape=(0, 299, 299, 3)
        )
        xlabel_storage = hdf5_file.create_earray(
            hdf5_file.root,
            "pos_xlabel",
            tb.UInt32Atom(),
            shape=(0, 1)
        )
        ylabel_storage = hdf5_file.create_earray(
            hdf5_file.root,
            "pos_ylabel",
            tb.UInt32Atom(),
            shape=(0, 1)
        )

        while 1:
            counter += 1
            if counter % 100 == 0:
                print("{} tiles saved in hdf5.".format(counter))
            data = q.get()
            if data == 'kill':
                print("Listner closed.")
                return None
            pred = data['pred']
            xlabel = data['xlabel']
            ylabel = data['ylabel']

            pred_storage.append(pred[None])
            xlabel_storage.append(xlabel[None])
            ylabel_storage.append(ylabel[None])

    finally:
        hdf5_file.close()


@timer
def slide_scn(scn_file, output_path, save_tiles=False):
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
        Tiles will be saved in the save folder if save_tiles flag is set to
        True.
    """

    # open scn_file
    try:
        scn_file = OpenSlide(scn_file)

    except OpenSlideUnsupportedFormatError:
        logging.error("OpenSlideUnsupportedFormatError!")
        return
    except OpenSlideError:
        logging.error("OpenSlideError!")

    # get attributes of the scn_file
    x0 = int(scn_file.properties["openslide.bounds-x"])
    y0 = int(scn_file.properties["openslide.bounds-y"])
    width = int(scn_file.properties["openslide.bounds-width"])
    height = int(scn_file.properties["openslide.bounds-height"])
    logging.debug(f'x0: {x0}, y0: {y0}, width: {width}, height: {height}')
    bound_x = x0 + width - 150
    bound_y = y0 + height - 150
    start_point = [x0 + 150, y0 + 150]
    scn_file.close()

    # create multiporcessing pool
    pool = mp.Pool(mp.cpu_count())
    manager = mp.Manager()
    q = manager.Queue()

    # run the listener
    watcher = mp.Process(target=listener, args=(q, output_path))

    # parameters passed to pool.map() function need to be packed in a list
    tasks = []
    task = {
            "bound_y": bound_y,
            "tile_path": Paths.data_test,
            "save_tiles": save_tiles,
            "queue": q
            }
    while start_point[0] < bound_x:
        task["start_point"] = list([start_point[0], start_point[1]])
        tasks.append(dict(task))
        start_point[0] += 150

    # slide images with multiprocessing
    jobs = []
    for task in tasks:
        job = pool.apply_async(v_slide, (task, ))
        jobs.append(job)

    watcher.start()
    for job in jobs:
        job.get()

    # kill listener
    q.put("kill")
    logging.debug("killer sent.")
    watcher.join()
    pool.close()
    pool.join()
    logging.debug("\nDone!")


if __name__ == "__main__":
    slide_scn(scn_file=sys.argv[1], output_path=sys.argv[2], save_tiles=False)
