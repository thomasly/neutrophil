'''
Filename: evaluation.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 4, 2018 5:12 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import os
import sys
import tables
import logging
import numpy as np
from keras.models import load_model
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
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


def plot_roc(fpr, tpr, roc_auc, output_filename):
    '''plot and save roc'''
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
    fig.savefig(output_filename, dpi=fig.dpi)
    logging.info(f'roc curve saved to {output_filename}')


def plot_prc(recall, precision, average_precision, output_filename):
    '''plot and save prc'''
    fig = plt.figure()
    step_kwargs = (
        {'step': 'post'}
        if 'step' in signature(plt.fill_between).parameters
        else {}
    )
    plt.step(
        recall, precision, color='b', alpha=0.2,
        where='post'
    )
    plt.fill_between(
        recall, precision, alpha=0.2, color='b', **step_kwargs
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    fig.savefig(output_filename, dpi=fig.dpi)
    logging.info(f'prc curve saved to {output_filename}')


def clear_plot():
    plt.clf()
    plt.cla()
    plt.close()


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
        filename_base = os.path.basename(h5_file).split('_')[0]

        hdf5_file = tables.open_file(pred_file, mode='r')
        m_pred = hdf5_file.root.test_img.shape[0]
        steps = int(ceil(m_pred / batch_size))
        generator = read_hdf5(hdf5_file, dataset="test", batch_size=32)

        preds = model.predict_generator(generator, steps=steps, verbose=1)
        preds = np.array(preds)[:, 1]
        logging.debug(f'preds: {preds}')
        true_values = hdf5_file.root.test_labels
        fpr, tpr, _ = roc_curve(list(true_values), list(preds))
        precision, recall, thresholds = prc(list(true_values), list(preds))
        average_precision = average_precision_score(
            list(true_values), list(preds))
        roc_auc = auc(fpr, tpr)

        roc_name = filename_base + "_roc.png"
        prc_name = filename_base + "_prc.png"
        plot_roc(fpr, tpr, roc_auc, roc_name)
        clear_plot()
        plot_prc(recall, precision, average_precision, prc_name)
        clear_plot()

    finally:
        hdf5_file.close()


def main():
    models = os.scandir(sys.argv[1])
    for model in models:
        logging.info(f'Analysing {model.name}')
        if not model.name.endswith('.hdf5'):
            logging.info(f'{model.name} is not a model file.')
            continue
        evaluate_model(model.path, sys.argv[2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
