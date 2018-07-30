########

# load_data_with_database.py

# use tensorflow.contrib.data.Database to load images data

#######

import tensorflow as  tf

NUM_CLASSES = 2

def input_parser(img_path, label):

	one_hot = tf.one_hot(label, NUM_CLASSES)

	img_file = tf.read_file(img_path)
	img_decoded = tf.image.decode_image(img_file, channels = 3)

	return img_decoded, one_hot


tr_data = tr_data.map(input_parser)

dataset = dataset.batch(batch_size)