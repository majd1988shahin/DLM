# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:27:47 2019

@author: khoefle
"""
# See: https://keras.io/datasets/#boston-housing-price-regression-dataset
from keras.datasets import boston_housing
from sklearn.ensemble import RandomForestClassifier

# 1) Try to understand the differences between training and testing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

y_train_class = [int(i) for i in y_train]


# 2) Look at a histogram of the data
import matplotlib.pyplot as plt
plt.close()
plt.plot(y_train_class)
results = []

hist=plt.hist(y_train)

# Try to understand what the criterion means, why is it mse? 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)

clf.fit(x_train,y_train_class)
    
result = clf.predict(x_test)

import numpy as np
# Write a function to describe the error
error = abs((y_test-result).mean())

#print(error)

from utils import metric

i1,i2=0,0
error=np.zeros(10)
N=100#dauert lang !!
while i1<10:
    #print(i1)
    while i2<N:
        np.random.seed(i2+13*i1+5)
        clf = RandomForestClassifier(n_estimators=100, max_depth=i1+1,random_state=0)
        clf.fit(x_train,y_train_class)
        result = clf.predict(x_test)
        error[i1]=error[i1]+metric(y_test,result)
        i2=i2+1
        #print(i2)
    i1=i1+1
    i2=0

error=error/N
plt.close()
plt.plot(np.arange(1,11),error)
print(error)



