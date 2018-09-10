import os, sys

class Paths:

    def __init__(self):

        # home directory
        self.source = os.path.dirname(os.path.abspath(__file__))
        self.home = os.path.dirname(self.source)

        # data directory
        self.data = os.path.join(self.home, "data")
        self.data_orig = os.path.join(self.data, "orig")
        self.data_pool = os.path.join(self.data, "pool")
        self.data_predict = os.path.join(self.data, "predict")
        self.data_test = os.path.join(self.data, "test")
        self.data_orig_76_79 = os.path.join(self.data_orig, "76_79")
        self.neg_76_79 = os.path.join(self.data_orig_76_79, "neg")
        self.pos_76_79 = os.path.join(self.data_orig_76_79, "pos")
        self.data_orig_80 = os.path.join(self.data_orig, "80")
        self.neg_80 = os.path.join(self.data_orig_80, "neg")
        self.pos_80 = os.path.join(self.data_orig_80, "pos")
        self.slice_80 = os.path.join(self.data_orig, "ImageCollection_80.scn")
        self.hdf5_76_79_80 = os.path.join(self.data, "76_79_80.hdf5")
        self.tiles_80 = os.path.join(self.data_test, "tiles_80.hdf5")

        # log directory
        self.logs = os.path.join(self.source, "logs")
        self.pred_cvs = os.path.join(self.data_test, "predictions.csv")

        # model directory
        self.models = os.path.join(self.source, "models")
