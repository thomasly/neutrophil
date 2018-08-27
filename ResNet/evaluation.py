import os, sys, tables
from tensorflow.keras.models import model_from_json
from math import ceil
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc

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
            inputs = hdf5_file.root.__getitem__(data_img)[n_start:n_end]
            yield inputs

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
        fpr, tpr, _ = roc_curve(list(true_values), list(preds))
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
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
        fig.savefig("aoc.png", dpi=fig.dpi)

    finally:
        hdf5_file.close()


if __name__ == "__main__":
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
