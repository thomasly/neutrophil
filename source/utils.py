'''
Filename: utils.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Oct 9, 2018 11:58 AM
-----
Last Modified: Oct 9, 2018 1:10 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import logging
from datetime import datetime
from functools import wraps


def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        value = fn(*args, **kwargs)
        end = datetime.now()
        timedelta = end - start
        logging.debug(f"Running time of {fn.__name__} is: {timedelta}")
        return value
    return wrapper


if __name__ == '__main__':
    from time import sleep
    logging.basicConfig(level=logging.DEBUG)

    @timer
    def say_hi():
        logging.debug("Hi")
        sleep(5)
        logging.debug("Hi after 5 seconds.")

    say_hi()
