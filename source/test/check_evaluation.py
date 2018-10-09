# import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG)
csv_file = open('tiles_80_preds.csv', 'r')

data = csv_file.readline()
data_list = []
while data:
    splitted = data.split(',')
    for i in splitted:
        data_list.append(float(i))
    data = csv_file.readline()
csv_file.close()
logging.info(f'# of data in data_list: {len(data_list)}')
logging.info(f'First data is: {data_list[0]}')
logging.info(f'Second data is: {data_list[1]}')
logging.info(f'Third data is: {data_list[2]}')

positive_counter = 0
negative_counter = 0
for idx in range(0, len(data_list), 2):
    if data_list[idx] > 0.5:
        negative_counter += 1
    else:
        positive_counter += 1
logging.info(f'# positive: {positive_counter}')
logging.info(f'# negative: {negative_counter}')
