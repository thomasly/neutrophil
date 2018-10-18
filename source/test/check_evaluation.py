import numpy as np
import logging
import tables as tb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
csv_file = open('epoch98_ResNet50_20181005_042643PM_tiles_76.csv', 'r')
hdf5_file = tb.open_file('../../data/test/tiles_76.hdf5', mode='r')

data = csv_file.readline()
logging.info(f'First line of data: {data}')
data_list = []
while data:
    first = data.split(',')[1]
    data_list.append(float(first))
    data = csv_file.readline()
csv_file.close()
logging.info(f'# of data in data_list: {len(data_list)}')
logging.info(f'First data is: {data_list[0]}')
logging.info(f'Second data is: {data_list[1]}')
logging.info(f'Third data is: {data_list[2]}')

positive_counter = 0
negative_counter = 0
for idx in range(len(data_list)):
    # if data_list[idx] < 0.5:
    #     negative_counter += 1
    if data_list[idx] > 0.5:
        continue
    # else:
    #     positive_counter += 1
    logging.info(f'Prediction value: {data_list[idx]}')
    img = np.array(hdf5_file.root.__getitem__('pred_img')[idx])
    plt.imshow(img)
    plt.show()

logging.info(f'csv file: {csv_file.name}')
logging.info(f'# positive: {positive_counter}')
logging.info(f'# negative: {negative_counter}')
