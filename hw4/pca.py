# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:39:47 2018

@author: Owner
"""

import sys
import numpy as np
import os
from os import listdir
from skimage import io
from skimage import transform


image_dir = sys.argv[1]
image_file = listdir(image_dir)

image = []

for i in image_file:
    old_img = io.imread(os.path.join(image_dir, i))
    #new_img = transform.resize(old_img, (200,200,3))
    #image.append(new_img)
    image.append(old_img)
    
X = np.reshape(image, (415,-1))
X_mean = np.mean(X, axis = 0)

print("SVD")
u, s, v = np.linalg.svd((X - X_mean).T, full_matrices = False)


target_name = sys.argv[2]
#img_target = io.imread(os.path.join(image_dir, target_name))
img_target = io.imread(os.path.join(image_dir, target_name)).flatten
#img_target = transform.resize(img_target, (200,200,3)).flatten()
img_target = img_target - X_mean

weight = np.dot(img_target, u[:, :4])

img_recon = np.dot(weight, u[:, :4].T) + X_mean
img_recon -= np.min(img_recon)
img_recon /= np.max(img_recon)
img_recon = (img_recon * 255).astype(np.uint8)
io.imsave("reconstruction.jpg", img_recon.reshape(600,600,3))
