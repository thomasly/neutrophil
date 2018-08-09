# -*- coding: utf-8 -*-

import pandas as pd
import os

def plot_loss():
    """
    """
    
    home_path = os.path.abspath(".")
    log_path = os.path.join(home_path, "batch_loss.log")
    
    df = pd.read_csv(log_path)
    ax = df.drop(['epoch', 'batch_total', 'batch', 'size'], axis=1).plot(secondary_y = ["acc"])
    ax.set_ylabel("Loss")
    ax.right_ax.set_ylabel("Acc")
    ax.set_xlim([-1, 21000])
#    ax.right_ax.set_ylim([-0.05, 0.6])
    
    
    log_path = os.path.join(home_path, "epoch_loss.log")
    
    df = pd.read_csv(log_path)
    ax = df.drop(['epoch'], axis=1).plot(secondary_y = ["acc", "val_acc"])
    ax.set_ylabel("Loss")
    ax.right_ax.set_ylabel("Acc")
#    ax.legend(loc = 'best')
    
if __name__ == "__main__":
    plot_loss()