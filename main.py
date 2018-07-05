from keras.datasets import mnist
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils

def mlp_model(input_shape, num_classes=10):
    """
    Multilayer perceptron model. Not a very deep network.
    :param input_dim:
    :param num_classes:
    :return:
    """
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2)) # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512, input_shape=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#     model = Sequential()
#     model.add(Dense(512, input_shape=(784,)))
#     model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
#                                   # of the layer above. Here, with a "rectified linear unit",
#                                   # we clamp all values below 0 to 0.

#     model.add(Dropout(0.2))   
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(10))
#     model.add(Activation('softmax')) # This special "softmax" activation among other things,
#                                      # ensures the output is a valid probaility distribution, that is
#                                      # that its values are all non-negative and sum to 1.



if __name__ == '__main__':
    print('-'*40+'reading data')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

#     im1 = np.array(x_train[:1]).reshape(28, 28)
#     cv2.imwrite('mnist_sample1.jpg', im1)

    print('Here is an example of the first image in training set...')
    cv2.imshow('mnist_sample1.jpg', im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    X_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')/255.
    X_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')/255.
    print('Training dataset shape: {}'.format(x_train.shape))
    print('Testing dataset shape: {}'.format(x_test.shape))
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    print('-'*40+'creating model')
    model = mlp_model((784, ), num_classes=10)
    print(model)
    model.fit(X_train, Y_train, epochs=10, batch_size=32)
    score = model.evaluate(X_test, Y_test)
    print("Our Test Accuracy was {}".format(score))
