
# coding: utf-8

# In[1]:


import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Concatenate, Dot, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.regularizers import l2
#from keras.utils import to_categorical


# In[2]:


movie_path = 'movies.csv'
user_path = 'users.csv'
train_path = 'train.csv'
test_path = 'test.csv'
batch = 512
epoch = 100


# In[3]:


def read_user(path):
    pass

def read_movie(path):
    all_gen = []
    movie_gen = []
    ind = []
    with open(path, 'r', encoding='latin-1') as f:
        f.readline()
        i = 0
        for line in f:
            tmp = []
            ind.append(line.split('::')[0])
            title = line.split('::')[1]
            genre = line.split('::')[2].strip().split('|')
            for g in genre:
                if g not in all_gen:
                    all_gen.append(g)
                tmp.append(all_gen.index(g))
            movie_gen.append(tmp)

    movie_num = len(movie_gen)
    gen_num = len(all_gen)
    matrix = np.zeros((movie_num, gen_num))
    for i in range(len(movie_gen)):
        for j in movie_gen[i]:
            matrix[i][j] = 1
    
    return all_gen, matrix  # len(18) / movie_num*gen_num
        

def read_train(path):
    train = []
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            train.append(line.strip().split(',')[1:4])
    
    t = np.array(train, dtype = int)
    np.random.shuffle(t)
    user_id = t[:, 0]
    movie_id = t[:, 1]
    rating = t[:, 2]
    
    return user_id, movie_id, rating


# In[4]:


def rmse(y, y_pred):
    r = K.sqrt( K.mean((y - y_pred)**2) )
    return r

#def rmse(y, y_pred):
#    r = np.sqrt(np.mean((y_pred - y)**2))
#    return r

def split_val(user_id, movie_id, rating, ratio):
    val_num = int(len(user_id) * ratio)
    u_val = user_id[0 : val_num, ]
    m_val = movie_id[0 : val_num, ]
    r_val = rating[0 : val_num, ]
    
    u = user_id[val_num :, ]
    m = movie_id[val_num :, ]
    r = rating[val_num : , ]
    
    return u, m, r, u_val, m_val, r_val


# In[5]:


all_gen, movie_gen = read_movie(movie_path)


# In[5]:


user_id, movie_id, rating = read_train(train_path)

#normalization rating
mean = np.mean(rating)
std = np.std(rating)
rating_norm = (rating - mean) / std

print(user_id.shape, movie_id.shape, rating.shape)


# In[6]:


print(mean)
print(std)


# In[19]:


user_num = np.max(user_id)
movie_num = np.max(movie_id)
emb_dim = 250


# In[8]:


# split validation
u_train, m_train, r_train, u_val, m_val, r_val = split_val(user_id, movie_id, rating_norm, 0.1)


# In[20]:


# MF model
input_user = Input(shape = (1,))
user_emb = Embedding(user_num, emb_dim, input_length = 1, embeddings_regularizer = l2(1e-5))(input_user)
user_vec = Flatten()(user_emb)

input_movie = Input(shape = (1,))
movie_emb = Embedding(movie_num, emb_dim, input_length = 1, embeddings_regularizer = l2(1e-5))(input_movie)
movie_vec = Flatten()(movie_emb)

#bias
user_b_emb = Embedding(user_num, 1, embeddings_regularizer = l2(1e-5))(input_user)
user_bias = Flatten()(user_b_emb)

movie_b_emb = Embedding(movie_num, 1, embeddings_regularizer = l2(1e-5))(input_movie)
movie_bias = Flatten()(movie_b_emb)

dot = Dot(axes=1)([user_vec, movie_vec])
out = Add()([user_bias, movie_bias, dot])

model = Model(inputs = [input_user, input_movie], outputs = dot)
model.summary()
model.compile(optimizer = 'adam', loss = 'mse', metrics = [rmse])


# In[21]:


#fit
filepath="weights.dim250.{epoch:02d}-{val_rmse:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_rmse', 
                             verbose=1, save_best_only=True)

early_stop = EarlyStopping(monitor = 'val_rmse', patience=5, verbose=1)
callbacks_list = [checkpoint, early_stop]
history = model.fit([u_train, m_train], r_train,
                batch_size = batch, 
                epochs = epoch,
                validation_data = ([u_val, m_val], r_val),
                callbacks = callbacks_list)

