# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:27:05 2019

@author: Kevin
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from PIL import
#(x_train, y_train), (x_test, y_test) =mnist.load_data()[:10]
x_test=np.load("x_test.npy")#(10000, 28, 28)
y_test=np.load("y_test.npy")
x_train=np.load("x_train.npy")#(60000, 28, 28)
y_train=np.load("y_train.npy")
# 3)
for i in range(0,10):
    plt.figure(i)
    plt.imshow(x_train[i],cmap='gray') 

# 4)
plt.figure(11)
y_train_hist_n=plt.hist(y_train)
plt.figure(12)
y_test_hist_n=plt.hist(y_test)

y_test   
# 5)
average_image = np.reshape(np.array([x_train[:,x,y].mean() 
    for x in range(0,28) for y in range(0,28)]),(28,28))
plt.figure(13)
plt.imshow(average_image,cmap='gray')
# 6)
for i in range(0,10):
    
    img = x_train[np.where(y_train==i)[0][0] ,:,:]#erses Bilde von Klasse i ist 
    img_average = img -average_image
        
    # get some image
    plt.figure(i)
    plt.imshow(img_average,cmap="gray")

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
