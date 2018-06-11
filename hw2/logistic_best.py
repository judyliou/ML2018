#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:07:40 2018

@author: ntueconfbra1
"""


import sys
import numpy as np
import pandas as pd
import csv


def sigmoid(z):
    s = 1 / (1.0 + np.exp(-z))
    return s

#load weight
w = np.load('model_log01_2.npy') 
b = np.load('model_log01_2b.npy')


#load data
dx_train, dy_train, dx_test = sys.argv[3], sys.argv[4], sys.argv[5]
x_train = pd.read_csv(dx_train).as_matrix() #shape: 32561x123
y_train = pd.read_csv(dy_train, header = None).as_matrix() #shape: 32561x1
y_train= y_train.reshape(y_train.shape[0]) #shape: (32651,)
x_test = pd.read_csv(dx_test).as_matrix() #shape: 16281x123


#select feature
simple = [ i for i in range(x_train.shape[1])]
square = [0, 10, 78, 79, 80]
cubic = [0, 10, 78, 79, 80]
x_train = np.concatenate((x_train[:, simple], 
                          x_train[:, square]**2,
                          x_train[:, cubic]**3), axis = 1)
x_test = np.concatenate((x_test[:,simple], 
                         x_test[:,square]**2,
                         x_test[:,cubic]**3), axis = 1)

#normalization
mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


#predict answer
ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    a = np.dot(x_test[i], w) + b
    s = sigmoid(a)
    if s > 0.5:
        ans[i].append(1)
    else:
        ans[i].append(0)
    
#output
filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close() 
