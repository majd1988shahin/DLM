# -*- coding: utf-8 -*-
"""Conv2D.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EIUgWdbUSqovP7wIZ18QbLAUkBORWxHN
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D ,MaxPooling1D ,Reshape
from keras.layers import Activation, Dropout, Flatten, Dense, Input

(x_train, y_train), (x_test, y_test) =mnist.load_data()
average_image = np.reshape(np.array([x_train[:,x,y].mean() 
    for x in range(0,28) for y in range(0,28)]),(28,28))

from keras.utils import to_categorical

y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)

x_train.shape

input1=Input(shape=(28,28))
x=Reshape((28,28,1))(input1)
x=Conv2D(32, (2, 2),activation="relu")(x)

x=Flatten()(x)
#x=MaxPooling1D(10)(x)
outputs=Dense(10,activation="softmax",name="output")(x)
model=Model(inputs=input1,outputs=outputs)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train_cat,validation_data=(x_test,y_test_cat),epochs=5)

