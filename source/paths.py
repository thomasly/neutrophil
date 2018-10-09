import os


class Paths:

    # home directory
    source = os.path.dirname(os.path.abspath(__file__))
    home = os.path.dirname(source)

    # data directory
    data = os.path.join(home, "data")
    data_orig = os.path.join(data, "orig")
    data_pool = os.path.join(data, "pool")
    data_predict = os.path.join(data, "predict")
    data_test = os.path.join(data, "test")
    data_orig_76_79 = os.path.join(data_orig, "76_79")
    neg_76_79 = os.path.join(data_orig_76_79, "neg")
    pos_76_79 = os.path.join(data_orig_76_79, "pos")
    data_orig_80 = os.path.join(data_orig, "80")
    neg_80 = os.path.join(data_orig_80, "neg")
    pos_80 = os.path.join(data_orig_80, "pos")
    slice_80 = os.path.join(data_orig, "ImageCollection_80.scn")
    hdf5_76_79_80 = os.path.join(data, "76_79_80.hdf5")
    tiles_80 = os.path.join(data_test, "tiles_80.hdf5")

    # log directory
    logs = os.path.join(source, "logs")

    # model directory
    models = os.path.join(source, "models")
