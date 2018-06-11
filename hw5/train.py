
# coding: utf-8

# In[1]:


import sys
import numpy as np
import collections
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import word2vec


# In[2]:


train_path = sys.argv[1]
nolabel_path = sys.argv[2]

val_ratio = 0.1
batch = 64
epoch = 10
vocab_size = 10000


# In[3]:


###I/O
def read_training_data(path):
    x_train, y_train = [], []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            tmp = line.strip().split(' +++$+++ ')
            x_train.append(tmp[1].split())
            y_train.append(tmp[0])
    return x_train, y_train

def read_nolabel_data(path):
    x = []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            x.append(line.strip().split())
    return x

def read_testing_data(path):
    x_test = []
    with open(path, 'r', encoding = 'utf8') as f:
        f.readline()
        for line in f:
            idx = line.find(',')
            x_test.append(line[idx+1:].strip().split())
    return x_test

def write_output(path, y_pred):
    with open(path, 'w', encoding='utf8') as f:
        f.write('id,label\n')
        for i in range(len(y_pred)):
            result = 0
            if y_pred[i] > 0.5:
                result = 1
            f.write(str(i) + ',' + str(result) + '\n')
    print('Output:', path)
    


# In[4]:


def split_val(x, y, ratio):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    #print(idx)
    num_val = int(x.shape[0] * ratio)
    val_idx = idx[:num_val]
    train_idx = idx[num_val:]
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, y_train, x_val, y_val

def val_accuracy(model, x_val, y_val):
    val_result = model.predict(x_val, batch_size = batch, verbose = 1)
    acc = 0
    for i in range(len(val_result)):
        if val_result[i] > 0.5:
            result = 1
        else: 
            result = 0
        if y_val[i] == result:
            acc += 1
    return (acc/len(x_val))


# In[5]:


## corpus preprocessing 
def preprocess(corpus): #trim + not letter
    corpus_new = []
    for sentence in corpus:
        sentence_new = []
        for word in sentence:
            new = ''
            i = 0
            while i < len(word):
                if check_ascii(word[i]):
                    new += word[i]
                    j = i + 1
                    while j < len(word) and word[j] == word[i]:
                        j += 1
                    i = j
                else:
                    i += 1
            if new != '':
                sentence_new.append(new)
        corpus_new.append(sentence_new)
    return corpus_new
                
def check_ascii(character):
    if (ord(character) >= 95 and ord(character) < 122):
        return True
    return False


# In[6]:


# word2index, word2vec
def build_freq_dict(text):
    freq = collections.Counter()
    for i in text:
        for j in i:
                freq[j] += 1
    return freq

def build_word2index_dict(freq_dict, size = vocab_size):
    word2index = {word[0]: i+2 for i, word in enumerate(freq_dict.most_common(size))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    return word2index

def convert2index(corpus, word2index):  ###### max_len ######
    corpus_idx = []
    #max = 0
    for sentence in corpus:
        seq = []
        for word in sentence:
            seq.append(word2index.get(word, 1))
        corpus_idx.append(seq)
        #if len(seq) > max:
            #max = len(seq)
    x = pad_sequences(corpus_idx, maxlen = 39)
    #print(x.shape[1], max)
    return x

def get_embedding_maxtrix(corpus, dim, word2index):
    # convert word to vector
    w2v = word2vec.Word2Vec(corpus, size = dim)
    index2word = {v: k for (k,v) in word2index.items()}
    # index2vector matrix
    matrix = np.zeros((len(index2word), dim))
    for i in range(1, len(index2word)):
        matrix[i] = w2v[index2word[i]]
    return matrix



# In[10]:


x_label, y_label = read_training_data(train_path)
x_nolabel = read_nolabel_data(nolabel_path)
#x_test = read_testing_data(test_path)


# In[11]:


print('preprocess')
x_label_1, x_nolabel_1 = preprocess(x_label), preprocess(x_nolabel)
all_text = x_label_1 + x_nolabel_1
print('x_label len:', len(x_label_1), 'y_label:', len(y_label))


# In[12]:


print('======= Build Freq Dict ========')
freq_dict = build_freq_dict(all_text)
print('len freq_dict:', len(freq_dict))
print('======= Build Wrod2index ========')
word2index = build_word2index_dict(freq_dict)
np.save('word2index_01.npy', word2index) 
print('len word2index:', len(word2index))
print('======= Convert to Index ========')
x = convert2index(x_label_1, word2index)


# In[13]:


# split validation set
print('Split Validation')
y_label = np.array((y_label), dtype = np.float32)
x_train, y_train, x_val, y_val = split_val(x, y_label, val_ratio)
print(x_train.shape)


# In[14]:


def filter2unk(all_text, word2index): 
    corpus = []
    for sentence in all_text:
        s = []
        for word in sentence:
            if word in word2index.keys():
                s.append(word)
            else:
                s.append('UNK')               
        corpus.append(s)
    return corpus

all_text_1 = filter2unk(all_text, word2index)


# In[15]:


# embedding
emb_matrix = get_embedding_maxtrix(all_text_1, 300, word2index)
emb_layer = Embedding(len(emb_matrix), output_dim = 300, weights=[emb_matrix], input_length = 39, trainable=False)


# In[16]:


# model
model = Sequential()
model.add(emb_layer)
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

filepath = "weights.0526.{epoch:02d}-{val_acc:.5f}.h5"
output_path = 'predict_0526_{val_acc:.5f}.csv'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                             verbose=1, save_best_only=True)

early_stop = EarlyStopping(monitor = 'val_acc', patience=1, verbose=1)
callbacks_list = [checkpoint, early_stop]
history = model.fit(x_train, y_train,
                batch_size = batch, 
                epochs = epoch,
                validation_data = (x_val, y_val),
                callbacks = callbacks_list)

