## generate_datasets.py

# Yang Liu

import os, sys, random
import shutil

def distribute_files(orig_path, train_ratio):
	"""
	"""
	home_path = os.path.abspath("..") + os.sep
	train_path = home_path + "data" + os.sep + "train" + os.sep
	test_path = home_path + "data" + os.sep + "test" + os.sep
	train_pos_path = train_path + "pos" + os.sep
	train_neg_path = train_path + "neg" + os.sep
	test_pos_path = test_path + "pos" + os.sep
	test_neg_path = test_path + "neg" + os.sep
	for path in [train_pos_path, train_neg_path, test_pos_path, test_neg_path]:
		if not os.path.exists(path):
			os.mkdir(path)

	pos_path = orig_path + "pos" + os.sep
	neg_path = orig_path + "neg" + os.sep
	pos_files = os.listdir(pos_path)
	neg_files = os.listdir(neg_path)

	n_pos_files = len(pos_files)
	n_pos_train = int(n_pos_files * train_ratio)
	n_iter = n_pos_train
	for _ in range(n_iter):
		index = random.randrange(n_pos_train)
		file_name = pos_files[index]
		file_path = pos_path + file_name
		shutil.copy(file_path, train_pos_path)
		n_pos_train -= 1
		pos_files.pop(index)
	for file in pos_files:
		file_path = pos_path + file
		shutil.copy(file_path, test_pos_path)

	n_neg_files = len(neg_files)
	n_neg_train = int(n_neg_files * train_ratio)
	n_iter = n_neg_train
	for _ in range(n_iter):
		index = random.randrange(n_neg_train)
		file_name = neg_files[index]
		file_path = neg_path + file_name
		shutil.copy(file_path, train_neg_path)
		n_neg_train -= 1
		neg_files.pop(index)
	for file in neg_files:
		file_path = neg_path + file
		shutil.copy(file_path, test_neg_path)
		


def generate_datasets(orig_folder, train_ratio):
	"""
	"""
	orig_path_base = os.path.abspath("..") + os.sep + "data" + os.sep + "orig" + os.sep
	orig_path = orig_path_base + orig_folder + os.sep

	distribute_files(orig_path, float(train_ratio))
	print "Done!"


def main():
	generate_datasets(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
	main()







