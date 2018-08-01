# -*- coding: utf-8 -*-


from keras.callbacks import Callback
import six, csv, os
import numpy as np
from collections import Iterable, OrderedDict

class LossHistory(Callback):
    """Callback that streams batch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
    ```python
    csv_logger = LossHistory('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(Callback, self).__init__()
        

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
        
    def on_epoch_end(self, epoch, logs=None):
        pass


    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + ['batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        n_batch = self.params['steps'] * self.epoch + batch
        row_dict = OrderedDict({'epoch': self.epoch, 'batch_total': n_batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None