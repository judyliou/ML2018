# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:28:15 2018

@author: Owner
"""
import sys
import csv 
import math
import random
import numpy as np

#DATA PARSING
data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('hw1_train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()


#GET TRAINING DATA
x = []
y = []

for i in range(12):
    for j in range(471):           #一個月取連續10小時的data可以有471筆
        x.append([])
        for t in range(18):        #總共有18種污染物
            for s in range(9):     #連續9小時
                x[471*i+j].append(data[t][480*i+j+s] ) #一個月有24*20=480hr
        y.append(data[9][480*i+j+9])



#PREPROCESSING!
#change odd values if too large or negative
# =============================================================================
# y_out = []
# for i in range(len(y)):
#     if y[i] > 200 or y[i] < 0:
#         y_out.append(i)
# 
# for i in range(len(x)):
#     for j in range(81,90):
#         if x[i][j] > 200 or x[i][j] < 0:
#             x[i][j] = np.nan
# 
# for i in y_out:
#     y[i] = np.nanmean(x[i][81:90])
# 
# for i in range(len(x)):
#     for j in range(81,90):
#         if np.isnan(x[i][j]) == True:
#             x[i][j] = np.nanmean(x[i][81:90])
# =============================================================================

out = []
for i in range(len(y)):
    if y[i] > 200 or y[i] < 0:
        out.append(i)

for i in range(len(x)):
# =============================================================================
#     for j in range(len(x[0])):
#         if x[i][j] < 0:
#              if i not in out:
#                 out.append(i)
#                 break
# =============================================================================
    for j in range(81,90):
        if x[i][j] > 200 or x[i][j] < 0:
            if i not in out:
                out.append(i)
                break

for i in sorted(out, reverse=True):
    del y[i]
    del x[i]


x = np.array(x)
y = np.array(y)


##SELECT FEATURE 
feature = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]  #1-18
if len(feature) > 0:
	x_f = x[:, (feature[0]-1)*9:feature[0]*9]
	for i in range(1,len(feature)):
		x_f = np.concatenate((x_f, x[:, (feature[i]-1)*9:feature[i]*9]), axis = 1)


#ADD BIAS
x_f = np.concatenate((np.ones((x_f.shape[0],1)),x_f), axis=1)

#seperate for testing
d = np.concatenate((y.reshape(-1,1),x_f), axis = 1)
random.shuffle(d)
random.shuffle(d)
random.shuffle(d)
x_train = d[:4800, 1:]
x_test = d[4800:, 1:]
y_train = d[:4800, 0]
y_test = d[4800:, 0]



#SET PARAMETERS
w = np.zeros(len(x_train[0]))         # initial weight vector
lr = 10                               # learning rate
iter = 100000                         # iteration

#ADAGRAD
x_t = x_train.transpose()
s_gra = np.zeros(len(x_train[0]))

for i in range(iter):
    hypo = np.dot(x_train,w)
    loss = hypo - y_train
    cost = np.sum(loss**2) / len(x_train)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra/ada
    if (i) % 10000 == 0:
        print ('iteration: {} | Cost: {}'.format(i, cost_a))


#validation
a = np.dot(x_test,w)
loss = a - y_test
cost = np.sum(loss**2) / len(x_test)
cost_a  = math.sqrt(cost)
print ('Test Cost: {}'.format(cost_a))



#SAVE/READ MODEL
# save model
np.save('model4.npy',w)
# read model
#w = np.load('model4.npy')


#GET TESTING DATA 
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
#text = open('hw1_test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)


#feature processing
if len(feature) > 0:
	test_f = test_x[:, (feature[0]-1)*9:feature[0]*9]
	for i in range(1,len(feature)):
		test_f = np.concatenate((test_f, test_x[:, (feature[i]-1)*9:feature[i]*9]), axis = 1)


#add bias
test_f = np.concatenate((np.ones((test_f.shape[0],1)),test_f), axis=1)


#PREDICT ANSWER
ans = []
for i in range(len(test_f)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_f[i])
    ans[i].append(a)

filename = sys.argv[2]
#filename = 'hw1_output_15.csv'
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()