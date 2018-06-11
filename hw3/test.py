# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:38:57 2018

@author: Owner
"""
import csv
import sys
import numpy as np
from keras.models import load_model

if sys.argv[3] == "public":
    try:
        model = load_model('model_19.h5?dl=1%0D')
    except:
        model = load_model('model_19.h5')
else:
    try:
        model = load_model('model_23.h5?dl=1%0D')
    except:
        model = load_model('model_23.h5')


batch = 128
mean = np.genfromtxt('mean.csv',delimiter=',')
std = np.genfromtxt('std.csv',delimiter=',')

#read test data
te = sys.argv[1]
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

#predict 
result = model.predict(x_test, batch_size = batch, verbose = 1)


#write output
wr = sys.argv[2]
with open(wr, 'w', encoding='big5') as f:
    f.write('id,label\n')
    for i in range(len(result)):
        predict = np.argmax(result[i])
        f.write(str(i) + ',' + str(predict) + '\n')

