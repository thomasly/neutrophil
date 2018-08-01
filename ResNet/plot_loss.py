# -*- coding: utf-8 -*-

import pandas as pd
import os

def plot_loss():
    """
    """
    
    home_path = os.path.abspath(".")
    log_path = os.path.join(home_path, "training.log")
    
    df = pd.read_csv(log_path)
    ax = df.drop(['epoch'], axis=1).plot(secondary_y = ["loss", "val_loss"])
    ax.set_ylabel("Accuracy")
    ax.right_ax.set_ylabel("Loss")
    ax.right_ax.set_ylim([-0.05, 0.6])
    
    
if __name__ == "__main__":
    plot_loss()