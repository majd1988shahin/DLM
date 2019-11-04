# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:45:41 2019

@author: Kevin
"""

import keras
import numpy as np

from keras import datasets
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Activation
from keras.utils import to_categorical

#(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
x_test=np.load("x_test_cifar100.npy")
y_test=np.load("y_test_cifar100.npy")
x_train=np.load("x_train_cifar100.npy")
y_train=np.load("y_train_cifar100.npy")
x_train_cat = to_categorical(x_train)
y_train_cat = to_categorical(y_train)
x_test_cat = to_categorical(x_test)
y_test_cat = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
model.summary()

hist = model.fit(x_train_cat,y_train_cat,validation_data=\
              (x_test_cat,y_test_cat),epochs=15)
   
plt.figure(999)
plt.plot(hist.history["loss"])
    
plt.title("Training Loss")
plt.figure(998)
plt.plot(hist.history["val_loss"])
plt.title("Validation Loss")
plt.figure(888)
plt.plot(hist.history["acc"])
plt.title("Trainings Accuracy")

plt.figure(887)
plt.plot(hist.history["val_acc"])
plt.title("Validation Accuracy")



