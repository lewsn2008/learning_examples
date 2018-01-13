#!/usr/bin/env python
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import numpy as np


class ModelCreator(object):
    def __init__(self, class_num):
        self.class_num = class_num
        
    def create_seq_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.class_num, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        return model


(x_train, y_train), (x_test, y_test) = mnist.load_data(path='./data')
print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

print y_train.shape
y_train = to_categorical(y_train, 10)
print y_train.shape
y_test = to_categorical(y_test, 10)

creator = ModelCreator(10)
model = creator.create_seq_model()
model.fit(x_train,
          y_train,
          batch_size=256,
          epochs=10,
          validation_data=(x_test, y_test))

