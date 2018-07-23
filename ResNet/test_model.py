from ResNetFromFile import load_model_from_json
import load_data as ld
import os
from keras.models import Model

def test_model():

	POS = "pos_76_79"
	NEG = "neg_76_79"

	model = load_model_from_json()
	test_data_path_base = ".." + os.sep + "testSets" + os.sep

	pos_path = test_data_path_base + POS
	neg_path = test_data_path_base + NEG

	_, X_test_orig, _, Y_test_orig = ld.load_data(
		pos_path, 
		neg_path, 
		max_pos = 1500,
		max_neg = 1500,
		test = True)

	X_test = X_test_orig / 255.0
	Y_test = Y_test_orig

	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	preds = model.evaluate(X_test, Y_test)
	print("Model loss: {}".format(preds[0]))
	print("Model accuracy: {}".format(preds[1]))

if __name__ == '__main__':
	test_model()



