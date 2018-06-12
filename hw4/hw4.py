# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:49:18 2018

@author: Owner
"""
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


img = np.load(sys.argv[1])

# PCA  
mean = np.mean(img / 255, axis=0)
img_n = (img / 255) - mean
pca = PCA(copy=False, n_components = 400, whiten = True, svd_solver = "full") 
img_pca = pca.fit_transform(img_n)

# K means
img_km_400 = KMeans(n_clusters = 2).fit_predict(img_pca)

# write
t = sys.argv[2]
test = pd.read_csv(t).as_matrix()

#PREDICT ANSWER
ans = []
for i in range(len(test)):
    ans.append([str(i)])
    if img_km_400[test[i][1]] == img_km_400[test[i][2]]:
        ans[i].append(1)
    else:
        ans[i].append(0)
    if i % 2000 == 0 :
        print("now: ", i)
    

filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["ID","Ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close() 
