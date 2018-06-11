
# coding: utf-8

# In[1]:


import sys
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# In[2]:


test_path = sys.argv[1]
word2index = np.load('word2index.npy').item()
model = load_model('weights.semi.02-0.82730.h5')
output_name = sys.argv[2]
mode = sys.argv[3]

################### MODEL #################
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
# =============================================================================

# In[3]:


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

def convert2index(corpus, word2index): 
    corpus_idx = []
    for sentence in corpus:
        seq = []
        for word in sentence:
            seq.append(word2index.get(word, 1))
        corpus_idx.append(seq)
    x = pad_sequences(corpus_idx, maxlen = 39)
    return x


# In[4]:


x_test = read_testing_data(test_path)
x_test_1 = preprocess(x_test)
x_test_idx = convert2index(x_test_1, word2index)
print('predict')
y_pred = model.predict(x_test_idx)
write_output(output_name, y_pred)

