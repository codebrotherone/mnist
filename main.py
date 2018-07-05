from keras.datasets import mnist
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

def mlp_model(num_classes=10):
	"""
	Multilayer perceptron model. Not a very deep network.
	:param input_dim:
	:param num_classes:
	:return:
	"""
	model = Sequential()
	model.add(Dense(10, input_dim=784, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



if __name__ == '__main__':
	print('-'*40+'reading data')
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	im1 = np.array(x_train[:1]).reshape(28, 28)
	cv2.imwrite('mnist_sample1.jpg', im1)
	cv2.imshow('mnist_sample1.jpg', im1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	print(x_train)
	print('Here is an example of the first image in training set...')

	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)

	# flatten 28*28 images to a 784 vector for each image
	num_pixels = x_train.shape[1] * x_train.shape[2]
	x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

	print('-'*40+'creating model')
	model = mlp_model(num_classes=10)
	print(model)
	model.fit(x_train, y_train, epochs=10, batch_size=32)
	score = model.evaluate(x_test, y_test)
	print("Our Test Accuracy was {}".format(score))
