# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:19:19 2018

@author: Owner
"""

import csv
import sys
import numpy as np
from keras.models import load_model

def read_model_list(filename):
    list = []
    f = open(filename, "r")
    row = csv.reader(f, delimiter = ",")
    for r in row:
        model_name = r[0]
        weight = r[1]
        list.append((model_name, weight))
    
    return list

# =============================================================================
# if sys.argv[3] == "public":
#     try:
#         model = load_model('model_19.h5?dl=1%0D')
#     except:
#         model = load_model('model_19.h5')
# else:
#     try:
#         model = load_model('model_23.h5?dl=1%0D')
#     except:
#         model = load_model('model_23.h5')
# 
# =============================================================================

batch = 128
mean = np.genfromtxt('mean.csv',delimiter=',')
std = np.genfromtxt('std.csv',delimiter=',')

#read test data
te = 'test.csv'
x_t = []
with open(te, 'r', encoding='big5') as f:
    n_row = 0
    for r in list(csv.reader(f))[1:]:
        x_t.append([float(x) for x in r[1].split()])
        n_row += 1
        if n_row % 3000 == 0:
            print('row:', n_row)
            
x_test = np.array(x_t)
x_test /= 255
x_test = (x_test - mean) / std
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

#ensemble
print("load model")
model_list = read_model_list("ensemble_model_list.csv")
model_num = len(model_list)
        
#total_weight = 0
model = load_model('model_18.h5') 
result = model.predict(x_test, batch_size = batch, verbose = 1)
for i in range(1, model_num):
    model = load_model(model_list[i][0]) 
    r = model.predict(x_test, batch_size = batch, verbose = 1)
    result += r
    print(i)
#result = np.divide(result, 4)


#write output
#wr = sys.argv[2]
wr = 'output_ensemble_1.csv'
with open(wr, 'w', encoding='big5') as f:
    f.write('id,label\n')
    for i in range(len(result)):
        predict = np.argmax(result[i])
        f.write(str(i) + ',' + str(predict) + '\n')
