
# coding: utf-8

# In[1]:

import sys
from keras.models import load_model
import numpy as np
import keras.backend as K


# In[2]:


def rmse(y, y_pred):
    r = K.sqrt( K.mean((y - y_pred)**2) )
    return r

def read_test(path):
    test = []
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            test.append(line.strip().split(','))
    test = np.array(test, dtype = int)
    #print(test[1])
    #idx = test[:, 0]
    u = test[:, 1]
    m = test[:, 2]
    return u, m

def write_output(path, y_pred):
    with open(path, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(len(y_pred)):
            f.write(str(i+1) + ',' + str(y_pred[i]) + '\n')
    print('Output:', path)


# In[4]:


test_path = sys.argv[1]
model = load_model('weights.norm.reg.03-0.75875.h5', custom_objects={'rmse': rmse})
output_name = sys.argv[2]


# In[5]:


u_test, m_test = read_test(test_path)

y_pred = model.predict([u_test, m_test])
print('y_pred shape:', y_pred.shape)

# In[6]:


#### NORM ######
mean = 3.5817120860388076
std = 1.116897661146206
y_pred_n = y_pred * std + mean
y_pred1 = np.clip(y_pred_n, 1, 5).reshape(-1,)

# In[8]:


write_output(output_name, y_pred1)

