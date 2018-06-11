# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 22:58:37 2018

@author: Owner
"""

import sys
import csv 
import math
import random
import numpy as np

w = np.load('model7.npy')

feature = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#GET TESTING DATA 
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
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



test_f = np.concatenate((np.ones((test_f.shape[0],1)),test_f), axis=1)
# 增加bias項  


#PREDICT ANSWER
ans = []
for i in range(len(test_f)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_f[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()