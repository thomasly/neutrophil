# -*- coding: utf-8 -*-

from load_model_from_file import load_model_from_json
import os, sys
from PIL import Image
import numpy as np

def load_img(path):
    """
    """
    
    img = np.array(Image.open(path))[:, :, 0:3]
    img = np.expand_dims(img, axis=0)
    return img


def main():
    """
    """
    
    home = os.path.join(os.path.abspath('..'), 'data', 'orig')
    img_path = os.path.join(home, sys.argv[1], sys.argv[2], sys.argv[3])
    model = load_model_from_json()
    model.compile('adam', 'binary_crossentropy')
    
    img = load_img(img_path)
    print("Image shape: {}".format(img))
    pred = model.predict(img)
    
    print("Prediction: {}".format(pred[0]))
    

if __name__ == '__main__':
    main()
    