# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:25:17 2019

@author: Kevin
"""

from keras.datasets import cifar10

cifar10_labels = ["airplane",
                  "automobile",
                  "bird",
                  "cat",
                  "deer",
                  "dog",
                  "frog",
                  "horse",
                  "ship",
                  "truck"]

x_train = ...
y_train = ...
x_test = ...
y_test = ...

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.layers import Input
import cv2
import numpy as np

resized_size = (150,150)

# resize ( https://keras.io/applications/#resnet )
# (May take some time...)
x_train = np.array([cv2.resize(i,resized_size) / 255 for i in x_train])
x_test = np.array([cv2.resize(i,resized_size) / 255 for i in x_test])


# create the base pre-trained model
input_tensor = Input(shape=(x_train.shape[1],
                            x_train.shape[2],
                            x_train.shape[-1]))

# Hier wird das Modell geladen
base_model = InceptionV3(input_tensor=input_tensor,weights=None, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x
x = Dense(1024, activation='relu')(x)
predictions = Dense(..., activation=...)(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=..., loss=...)

model.fit(....,...,val_data=....,epochs=....)

# do the prediction on the test data
res = model.predict(...)

# convert back to labels
results_as_labels = np.argmax(...)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(..., ...)

from matplotlib import pyplot as plt
plt.imshow(...)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(...)
plt.yticks(...)

