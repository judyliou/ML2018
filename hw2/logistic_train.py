#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:56:29 2018

@author: ntueconfbra1
"""
import sys
import numpy as np
import pandas as pd
import csv


def sigmoid(z):
    s = 1 / (1.0 + np.exp(-z))
    return s


#load data
dx_train, dy_train, dx_test = sys.argv[3],sys.argv[4], sys.argv[5]
x_train = pd.read_csv(dx_train).as_matrix() #shape: 32561x123
y_train = pd.read_csv(dy_train, header = None).as_matrix() #shape: 32561x1
y_train= y_train.reshape(y_train.shape[0]) #shape: (32651,)


#select feature
simple = [ i for i in range(x_train.shape[1])]
square = [0, 10, 78, 79, 80]
cubic = [0, 10, 78, 79, 80]
x_train = np.concatenate((x_train[:, simple], 
                          x_train[:, square]**2,
                          x_train[:, cubic]**3), axis = 1)

#normalization
mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train - mean) / std
          

#initialization
b = 0.0
w = np.random.rand(x_train.shape[1]) * 0.1
lr = 0.1
epoch = 20000
reg = 0.0

w_gra_sum = np.zeros(x_train.shape[1])
b_gra_sum = 0


for i in range(epoch):
        x_train_mb, y_train_mb = x_train, y_train
        z = np.dot(x_train_mb, w) + b
        y = sigmoid(z)
        error = y_train_mb - y
        loss = (-(np.dot(y_train_mb, np.log(y)) + np.dot((1- y_train_mb), np.log(1 - y))) +\
            (reg/2) * np.dot(w, w)) / len(x_train_mb)
    
        w_grad = np.dot(-(y_train_mb - y).T,x_train_mb) + reg * w
        b_grad = np.sum(-(y_train_mb - y))
    
        w_gra_sum += w_grad**2
        b_gra_sum += b_grad**2
    
        w -= lr/np.sqrt(w_gra_sum) * w_grad
        b -= lr/np.sqrt(b_gra_sum) * b_grad
    
        #print
        if (i+1) % 1000 == 0:
            y[y >= 0.5] = 1
            y[y < 0.5] = 0
            acc = y_train_mb - y
            acc[acc ==  0] = 2
            acc[acc != 2] = 0
            print('epoch:{} | Loss:{} | Accuracy:{}\n'.format(i+1, loss, np.sum(acc)/(2*acc.shape[0])))


#=======================================================================
#test set
x_test = pd.read_csv(dx_test).as_matrix() #shape: 16281x123
x_test = np.concatenate((x_test[:,simple], 
                         x_test[:,square]**2,
                         x_test[:,cubic]**3), axis = 1)
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
    

filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close() 
