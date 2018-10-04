'''
Filename: evaluation.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 4, 2018 12:48 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import sys
import tables
from keras.models import load_model
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
plt.switch_backend('agg')


def read_hdf5(hdf5_file, dataset, batch_size):
    """
    """

    data_img = dataset + "_img"

    m_data = hdf5_file.root.__getitem__(data_img).shape[0]
    batch_list = list(range(int(ceil(m_data / batch_size))))

    while True:
        for num in batch_list:
            n_start = num * batch_size
            n_end = min((num + 1) * batch_size, m_data)
            inputs = hdf5_file.root.__getitem__(data_img)[n_start:n_end]
            yield inputs


def evaluate_model(h5_file, pred_file):
    """
    evaluate the trained model. Plot ROC curve and calculate AUC.

    inputs:
    model json file path, model weights file.

    outputs:
    filename of the plotting.
    """
    try:
        batch_size = 32
        model = load_model(h5_file)

        hdf5_file = tables.open_file(pred_file, mode='r')
        m_pred = hdf5_file.root.test_img.shape[0]
        steps = int(ceil(m_pred / batch_size))
        generator = read_hdf5(hdf5_file, dataset="test", batch_size=32)

        preds = model.predict_generator(generator, steps=steps, verbose=1)
        true_values = hdf5_file.root.test_labels
        fpr, tpr, _ = roc_curve(list(true_values), list(preds))
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        lw = 2
        plt.plot(
            fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Neutrophil Identifier ROC')
        plt.legend(loc="lower right")
        fig.savefig("aoc.png", dpi=fig.dpi)

    finally:
        hdf5_file.close()


if __name__ == "__main__":
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
