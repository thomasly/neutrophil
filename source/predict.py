'''
Filename: predict.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 9, 2018 11:44 AM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import os
import sys
import logging
from math import ceil
from load_model_from_file import load_model_from_json
import numpy as np
import tables as tb
from paths import Paths


def read_hdf5(hdf5_file, dataset="pred", batch_size=32):
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


def predict(model_path, hdf5_file_path=None):
    """
    """
    try:
        BATCH_SIZE = 32
        model = load_model_from_json()
        default_path = Paths.tiles_80
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

        save_path = os.path.join(Paths.data_test, "tiles_80_preds.csv")
        np.savetxt(save_path, preds, delimiter=',')
    except Exception as e:
        hdf5_file.close()
        logging.debug(e.with_traceback())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    predict("./models/{}".format(sys.argv[1]))
