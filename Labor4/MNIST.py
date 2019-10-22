# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:27:05 2019

@author: Kevin
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = ... # 2)

# 3)
for i in range(....):
    plt.figure(....)
    plt.imshow(....) 

# 4)
plt.figure(500)
plt.hist(...)
plt.figure(501)
plt.hist(...)
    
# 5)
average_image = ....

# 6)
for i in range(0,10):
    
    img = ...
    img_average = img -average_image
        
    # get some image
    plt.figure(i)
    plt.imshow(...)

    plt.figure(i+100)
    plt.hist(img_average.flatten())
    plt.figure(i+200) 
    plt.hist(img.flatten())
    
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import keras.backend as K
import gc

#7)
flat_input_train = np.reshape(x_train,(len(x_train),-1))
flat_input_train_minus_averaged = ...

flat_input_test = ...
flat_input_test_minus_averaged = ...

#8)
def create_model():
    
    model = Sequential()
    
    model.add(Dense(32, input_dim=flat_input_train.shape[1]))
    model.add(Dense(64,activation="relu"))
    
    # Final layer - choose the amount of classes
    model.add(Dense(10,activation="softmax"))
    return model

# 9)
y_train_cat = ...
y_test_cat = ...

# 10)
optimizers_to_test = ["rmsprop"]
for optimizer in optimizers_to_test:
    model = create_model()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = model.fit(flat_input_train,y_train_cat,validation_data=\
              (flat_input_test,y_test_cat),epochs=15)
    
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

    del hist
    del model
    K.clear_session()
    gc.collect()

plt.figure(999)
plt.legend(optimizers_to_test)
plt.figure(998)
plt.legend(optimizers_to_test)
plt.figure(888)
plt.legend(optimizers_to_test)
plt.figure(887)
plt.legend(optimizers_to_test)

# 11)
for optimizer in optimizers_to_test:
    model = create_model()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = ....
    
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

    del hist
    del model
    K.clear_session()
    gc.collect()

new_legend = optimizers_to_test + [i + " mean" for i in optimizers_to_test]
plt.figure(999)
plt.legend(new_legend)
plt.figure(998)
plt.legend(new_legend)
plt.figure(888)
plt.legend(new_legend)
plt.figure(887)
plt.legend(new_legend)
