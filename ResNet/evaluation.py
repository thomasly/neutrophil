import os, sys, tables
from tensorflow.keras.models import model_from_json
from math import ceil
from read_hdf5 import read_hdf5
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_model(json_file, weights, pred_file):
    """
    evaluate the trained model. Plot ROC curve and calculate AUC.

    inputs:
    model json file path, model weights file.

    outputs:
    filename of the plotting.
    """
    try:
        batch_size = 32
        with open(json_file, 'r') as f:
            json = f.read()
        model = model_from_json(json)
        model.load_weights(weights)

        hdf5_file = tables.open_file(pred_file, mode = 'r')
        m_pred = hdf5_file.root.val_img.shape[0]
        steps = int(ceil(m_pred / batch_size))
        generator = read_hdf5(hdf5_file, dataset="val", batch_size=32)

        preds = model.predict_generator(generator, steps=steps, verbose=1)
        true_values = hdf5_file.root.val_labels

        fpr, tpr, _ = roc_curve(true_values, preds)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Neutrophil Identifier ROC')
        plt.legend(loc="lower right")
        plt.show()

    finally:
        hdf5_file.close()


if __name__ == "__main__":
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
