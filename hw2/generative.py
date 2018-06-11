#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:20:55 2018

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
dx_train, dy_train = sys.argv[3], sys.argv[4]
x_train = pd.read_csv(dx_train).as_matrix() #shape: 32561x123
y_train = pd.read_csv(dy_train, header = None).as_matrix() #shape: 32561x1
y_train= y_train.reshape(y_train.shape[0]) #shape: (32651,)


#feature selection
x_train = np.delete(x_train, np.s_[11:27], axis=1)   


#normalization
mean = np.mean(x_train, axis = 0)
sd = np.std(x_train, axis = 0)
x_train = (x_train - mean) / sd
 
    
#calculate mu
n = x_train.shape[0]
m = x_train.shape[1]
mu1 = np.zeros((m,))
mu2 = np.zeros((m,))
count1 = 0
count2 = 0
for i in range(n):
    if y_train[i] == 1:
        mu1 += x_train[i]
        count1 += 1
    else:
        mu2 += x_train[i]
        count2 += 1
mu1 /= float(count1)
mu2 /= float(count2)


#calculate sigma
sigma1 = np.zeros((m, m))        
sigma2 = np.zeros((m, m))
for i in range(n):
    if y_train[i] == 1:
        sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
    else:
        sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
sigma1 /= float(count1)
sigma2 /= float(count2)
sigma_share = (float(count1) / n) * sigma1 + (float(count2) / n) * sigma2
 
             
#predict
def predict(x_test, mu1, mu2, sigma_share, count1, count2):
    sigma_inverse = np.linalg.inv(sigma_share)
    w = np.dot((mu1 - mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) +\
         0.5 * np.dot(np.dot([mu2], sigma_inverse), mu2) + \
         np.log(float(count1) / count2)
    x = x_test.T
    z = np.dot(w,x) + b
    y = sigmoid(z)
    return y  

y = predict(x_train, mu1, mu2, sigma_share, count1, count2)  
y1 = np.around(y)     
result = (y1 == y_train)
#print('Accuarcy:', result.sum()/result.shape[0])
              
#=================================================
#test set
dx_test = sys.argv[5]
x_test = pd.read_csv(dx_test).as_matrix() #shape: 16281x123
x_test = np.delete(x_test, np.s_[11:27], axis=1)  
x_test = (x_test - mean) / sd
 

#predict answer
ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    s = predict(x_test[i], mu1, mu2, sigma_share, count1, count2)
    if s >= 0.5:
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
