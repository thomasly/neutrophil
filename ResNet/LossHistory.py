# -*- coding: utf-8 -*-


from tensorflow.keras.callbacks import Callback
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

    def __init__(self, epoch_filename, batch_filename, model_file, weights_file, separator=',', append=False):
        self.sep = separator
        self.epoch_filename = epoch_filename
        self.batch_filename = batch_filename
        self.model_file = model_file
        self.weights_file = weights_file
        self.append = append
        self.batch_writer = None
        self.epoch_writer = None
        self.batch_keys = None
        self.epoch_keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(LossHistory, self).__init__()
        

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.epoch_filename):
                with open(self.epoch_filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.epoch_csv_file = open(self.epoch_filename, 'a' + self.file_flags)
        else:
            self.epoch_csv_file = open(self.epoch_filename, 'w' + self.file_flags)
            
        if self.append:
            if os.path.exists(self.batch_filename):
                with open(self.batch_filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.batch_csv_file = open(self.batch_filename, 'a' + self.file_flags)
        else:
            self.batch_csv_file = open(self.batch_filename, 'w' + self.file_flags)
            
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.epoch_keys is None:
            self.epoch_keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.epoch_keys])

        if not self.epoch_writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.epoch_writer = csv.DictWriter(self.epoch_csv_file,
                                         fieldnames=['epoch'] + self.epoch_keys, dialect=CustomDialect)
            if self.append_header:
                self.epoch_writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.epoch_keys)
        self.epoch_writer.writerow(row_dict)
        self.epoch_csv_file.flush()
        
        model_to_json = self.model.to_json()
        with open(self.model_file, "w") as f:
            f.write(model_to_json)
        print("Model saved to {}!".format(self.model_file))
        self.model.save_weights(self.weights_file)
        print("Model weights saved to {}".format(self.weights_file))
        

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

        if self.batch_keys is None:
            self.batch_keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.batch_keys])

        if not self.batch_writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.batch_writer = csv.DictWriter(self.batch_csv_file,
                                         fieldnames=['epoch'] + ['batch_total'] + self.batch_keys, dialect=CustomDialect)
            if self.append_header:
                self.batch_writer.writeheader()
        
        n_batch = self.params['steps'] * self.epoch + batch
        row_dict = OrderedDict({'epoch': self.epoch, 'batch_total': n_batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.batch_keys)
        self.batch_writer.writerow(row_dict)
        self.batch_csv_file.flush()
        

    def on_train_end(self, logs=None):
        self.epoch_csv_file.close()
        self.batch_csv_file.close()
        self.epoch_writer = None
        self.batch_writer = None
        