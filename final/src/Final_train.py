
# coding: utf-8

# In[1]:


# coding: utf-8
import numpy as np
from gensim.models import word2vec
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.layers import Bidirectional, Input, BatchNormalization, GRU, Activation, Dot
from keras.preprocessing import sequence  
from keras.layers import Concatenate, concatenate
import re
import csv
from collections import Counter
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(5566)


# In[2]:

def preprocess(string):
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    
    return string

def build_counter(texts):
    frequency = Counter()
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def valid_generator(data, label):
    tmp_data = []
    tmp_label = []
    randomlist = np.random.permutation(len(data))
    for i in range(len(data)):
        idx = randomlist[i]
        tmp_data.append(data[idx])
        tmp_label.append(label[idx])
    
    val_size = 20000
    tmp_data = np.array(tmp_data)
    tmp_label = np.array(tmp_label)
    
    return tmp_data[val_size:], tmp_label[val_size:], tmp_data[:val_size], tmp_label[:val_size]
def build_w2i_map(fre_dic, threshold):
    MAX_FEATURES = threshold
    
    word_freqs = fre_dic
    word2index = {'PAD':0}
    word2index.update({x[0]: i+1 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))})
    index2word = {v:k for k, v in word2index.items()}
    return word2index, index2word

def doc2seq(data, word2index, maxlen):
    X = np.empty(len(data),dtype=list)
    i=0
    for words in data:
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])            
        X[i] = seqs
        i += 1
    X = sequence.pad_sequences(X, maxlen)
    return X


def build_EmbeddingMatrix(sentence, index2word, pick_size):
    w2v_model = word2vec.Word2Vec(sentence, size=pick_size, window=5, min_count=0, workers=8)
    EM = np.zeros((len(index2word) ,pick_size))
    for i in range(1, len(index2word)):
        EM[i] = w2v_model[index2word[i]]
    return EM

def bag1sentence(doc):
    bag = []
    ans = []
    for i in range(len(doc) - 1):
        bag.append(doc[i])
        ans.append(doc[i + 1])
    return bag, ans

def bag2sentence(doc):
    bag = []
    ans = []
    for i in range(len(doc) - 2):
        bag.append(doc[i] + doc[i + 1])
        ans.append(doc[i + 2])
    return bag, ans

def bag3sentence(doc):
    bag = []
    ans = []
    for i in range(len(doc) - 3):
        bag.append(doc[i] + doc[i + 1] + doc[i + 2])
        ans.append(doc[i + 3])
    return bag, ans

def txt2QA(doc):
#     a, b = bag1sentence(doc)
    c, d = bag2sentence(doc)
    e, f = bag3sentence(doc)
#     return (a + c + e), (b + d + f)
    return (c + e), (d + f)

def read_test_data(path):
    Q, A = [], []
    with open(path, 'r', encoding = 'utf8') as f:
        f.readline()
        for line in f:
            #print(line)
            tmp = line.split(',')
            Q_sentence = tmp[1].strip(' ').split(' ')
            s = []
            for i in Q_sentence:
                s.append(i)
            Q.append(s)
            
            choices = tmp[2].split(':')
            #print(len(choices))
            for i in range(6):
                c1 = choices[i+1].split(' ')
                t = []
                for j in range(len(c1)-3):
                    t.append(c1[j+1])
                A.append(t)  #total: 6xdata.shape[0]
    return Q, A


def data_generator(data_Q1, data_A1, data_Q2, data_A2, batch_size, ratio = 5): 
    seq_len = len(data_Q2[0])
    data1 = np.concatenate((np.array(data_Q1), np.array(data_A1)), axis=1)
    data2 = np.concatenate((np.array(data_Q2), np.array(data_A2)), axis=1)

    while True:
        np.random.shuffle(data1)
        np.random.shuffle(data2)
        
        Q_tmp1 = data1[:, :seq_len]
        A_tmp1 = data1[:, seq_len:]
        
        Q_tmp2 = data2[:, :seq_len]
        A_tmp2 = data2[:, seq_len:]
        
        Q1_c, A1_c, L1_c = label_sentence(Q_tmp1, A_tmp1, ratio)
        Q2_c, A2_c, L2_c = label_sentence(Q_tmp2, A_tmp2, ratio)
        
        for i in range(int( (len(data1) + len(data2)) / batch_size)):
            per_bag = int(batch_size / 2)
            
            Q1 = Q1_c[i * per_bag : (i + 1) * per_bag, :]
            A1 = A1_c[i * per_bag : (i + 1) * per_bag, :]
            L1 = L1_c[i * per_bag : (i + 1) * per_bag, :]
            
            Q2 = Q2_c[i * per_bag : (i + 1) * per_bag, :]
            A2 = A2_c[i * per_bag : (i + 1) * per_bag, :]
            L2 = L2_c[i * per_bag : (i + 1) * per_bag, :]

            Q = np.concatenate((Q1, Q2), axis=0)
            A = np.concatenate((A1, A2), axis=0)
            L = np.concatenate((L1, L2), axis=0)

            yield(([Q] + np.hsplit(A, 6)), L)

            
def label_sentence(data_Q, data_A, ratio = 5): #ratio=5
    total = len(data_Q)
    ans_num = ratio + 1
    
    Q = data_Q
    ans = []
    label = []
    for i in range(total):
        correct = np.random.randint(ans_num)
        for j in range(ans_num):
            if j == correct:
                ans.append(data_A[i, :])
                label.append(1)
            else:
                ans.append(data_A[np.random.randint(total), :])
                label.append(0)
    A = np.array(ans).reshape(total, -1)
    L = np.array(label).reshape(total, ans_num)
                       
    return(Q, A, L)

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)


# In[3]:


sentence = []
pick_num = 3
fp = open("data/1_train.txt.seg", "r", encoding = 'utf-8')
txt1 = []
for line in fp.readlines():
    tmp = line.strip().split()
    if (len(tmp) > pick_num):
        txt1.append(tmp)
fp.close()

txt2 = []
fp = open("data/2_train.txt.seg", "r", encoding = 'utf-8')
for line in fp.readlines(): 
    tmp = line.strip().split()
    if (len(tmp) > pick_num):
        txt2.append(tmp)
fp.close()

txt3 = []
fp = open("data/3_train.txt.seg", "r", encoding = 'utf-8')
for line in fp.readlines(): 
    tmp = line.strip().split()
    if (len(tmp) > pick_num):
        txt3.append(tmp)
fp.close()

txt4 = []
fp = open("data/4_train.txt.seg", "r", encoding = 'utf-8')
for line in fp.readlines(): 
    tmp = line.strip().split()
    if (len(tmp) > pick_num):
        txt4.append(tmp)
fp.close()

txt5 = []
fp = open("data/5_train.txt.seg", "r", encoding = 'utf-8')
for line in fp.readlines(): 
    tmp = line.strip().split()
    if (len(tmp) > pick_num):
        txt5.append(tmp)
fp.close()

Q1,A1 = bag2sentence(txt1)
Q2,A2 = bag2sentence(txt2)
Q3,A3 = bag2sentence(txt3)
Q4,A4 = bag2sentence(txt4)
Q5,A5 = bag2sentence(txt5)

Q_bag2 = Q1 + Q2 + Q3 + Q4 + Q5
A_bag2 = A1 + A2 + A3 + A4 + A5

Q1,A1 = bag3sentence(txt1)
Q2,A2 = bag3sentence(txt2)
Q3,A3 = bag3sentence(txt3)
Q4,A4 = bag3sentence(txt4)
Q5,A5 = bag3sentence(txt5)

Q_bag3 = Q1 + Q2 + Q3 + Q4 + Q5
A_bag3 = A1 + A2 + A3 + A4 + A5

pre_sentences, next_sentences = read_test_data('data/testing_data.csv')
tmp = []
for sen in pre_sentences:
    for i in range(6):
        tmp.append(sen)
re_sentences = tmp

sentence = Q_bag2 + Q_bag3 + A_bag2 + A_bag3 + pre_sentences + next_sentences


# In[4]:


txt_counter = build_counter(sentence)
word2index, index2word = build_w2i_map(txt_counter, len(txt_counter))


# In[11]:


l = 10
for i in A_bag2:
    if(len(i) > l):
        l = len(i)
print(l)


# In[5]:

Q_bag2_seq = doc2seq(Q_bag2, word2index, 45)
A_bag2_seq = doc2seq(A_bag2, word2index, 15)
Q_bag3_seq = doc2seq(Q_bag3, word2index, 45)
A_bag3_seq = doc2seq(A_bag3, word2index, 15)

Q_bag2_seq, A_bag2_seq, val_Q1, val_A1 = valid_generator(Q_bag2_seq, A_bag2_seq)
Q_bag3_seq, A_bag3_seq, val_Q2, val_A2 = valid_generator(Q_bag3_seq, A_bag3_seq)
val_Q1, val_A1, l1 = label_sentence(val_Q1, val_A1, 5)
val_Q2, val_A2, l2 = label_sentence(val_Q2, val_A2, 5)
val_Q = np.concatenate((val_Q1, val_Q2), axis=0)
val_A = np.concatenate((val_A1, val_A2), axis=0)
val_L = np.concatenate((l1, l2), axis=0)

val_data = [val_Q] + np.hsplit(val_A, 6)


# In[7]:

w2v_matrix = build_EmbeddingMatrix(sentence, index2word, 200)
np.save('w2v.npy',w2v_matrix)
w2v_matrix = np.load('w2v.npy')


# In[8]:


embedding_layer1 = Embedding(len(w2v_matrix), output_dim = 200, weights=[w2v_matrix], input_length = 45, trainable=False)
embedding_layer2 = Embedding(len(w2v_matrix), output_dim = 200, weights=[w2v_matrix], input_length = 15, trainable=False)


# In[59]:


embedding_layer = Embedding(len(w2v_matrix), output_dim = 200, weights=[w2v_matrix], input_length = 45, trainable=False)
maxLen = 45
input_context = Input(shape = (maxLen, ), dtype = 'int32')
input_target1 = Input(shape = (maxLen, ), dtype = 'int32')
input_target2 = Input(shape = (maxLen, ), dtype = 'int32')
input_target3 = Input(shape = (maxLen, ), dtype = 'int32')
input_target4 = Input(shape = (maxLen, ), dtype = 'int32')
input_target5 = Input(shape = (maxLen, ), dtype = 'int32')
input_target6 = Input(shape = (maxLen, ), dtype = 'int32')

auto_model = Sequential()
auto_model.add(embedding_layer)
auto_model.add(Bidirectional(GRU(128,activation="tanh",dropout=0.2,return_sequences = True, kernel_initializer='he_uniform')))
auto_model.add(Bidirectional(GRU(64,activation="tanh",dropout=0.2,return_sequences = False, kernel_initializer='he_uniform')))

input_ctx_embed = auto_model(input_context)
input_tar_embed1 = auto_model(input_target1)
input_tar_embed2 = auto_model(input_target2)
input_tar_embed3 = auto_model(input_target3)
input_tar_embed4 = auto_model(input_target4)
input_tar_embed5 = auto_model(input_target5)
input_tar_embed6 = auto_model(input_target6)

DNN_model = Sequential()
DNN_model.add(Dense(input_shape = (1, 128) , units = 128, activation = 'relu'))
DNN_model.add(Dense(units = 128, activation = 'relu'))
DNN_model.add(Dense(units = 128, activation = 'relu'))

sim_tar_embed = DNN_model(input_ctx_embed)

out1 = Dot(axes = 1, normalize = False)([input_tar_embed1, sim_tar_embed])
out2 = Dot(axes = 1, normalize = False)([input_tar_embed2, sim_tar_embed])
out3 = Dot(axes = 1, normalize = False)([input_tar_embed3, sim_tar_embed])
out4 = Dot(axes = 1, normalize = False)([input_tar_embed4, sim_tar_embed])
out5 = Dot(axes = 1, normalize = False)([input_tar_embed5, sim_tar_embed])
out6 = Dot(axes = 1, normalize = False)([input_tar_embed6, sim_tar_embed])

out = Activation('softmax')(concatenate([out1, out2, out3, out4, out5, out6]))

Final_model = Model(inputs = [input_context, input_target1, input_target2, input_target3, input_target4, input_target5, input_target6], 
                    outputs = out)
Final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[25]:


get_custom_objects().update({'swish': Swish(swish)})
maxLen = 15
input_context = Input(shape = (45, ), dtype = 'int32')
input_target1 = Input(shape = (maxLen, ), dtype = 'int32')
input_target2 = Input(shape = (maxLen, ), dtype = 'int32')
input_target3 = Input(shape = (maxLen, ), dtype = 'int32')
input_target4 = Input(shape = (maxLen, ), dtype = 'int32')
input_target5 = Input(shape = (maxLen, ), dtype = 'int32')
input_target6 = Input(shape = (maxLen, ), dtype = 'int32')

tar_model = Sequential()
tar_model.add(embedding_layer2)
tar_model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True, kernel_initializer='he_uniform')))
tar_model.add(Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False, kernel_initializer='he_uniform')))
input_tar_embed1 = tar_model(input_target1)
input_tar_embed2 = tar_model(input_target2)
input_tar_embed3 = tar_model(input_target3)
input_tar_embed4 = tar_model(input_target4)
input_tar_embed5 = tar_model(input_target5)
input_tar_embed6 = tar_model(input_target6)

txt_model = Sequential()
txt_model.add(embedding_layer1)
txt_model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True, kernel_initializer='he_uniform')))
txt_model.add(Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False, kernel_initializer='he_uniform')))
input_ctx_embed = txt_model(input_context)

out1 = Concatenate()([input_tar_embed1, input_ctx_embed])
out2 = Concatenate()([input_tar_embed2, input_ctx_embed])
out3 = Concatenate()([input_tar_embed3, input_ctx_embed])
out4 = Concatenate()([input_tar_embed4, input_ctx_embed])
out5 = Concatenate()([input_tar_embed5, input_ctx_embed])
out6 = Concatenate()([input_tar_embed6, input_ctx_embed])
# print(out1.shape)
DNN_model = Sequential()
DNN_model.add(Dense(input_shape = (1, 256) , units = 128, activation = swish))
DNN_model.add(Dropout(0.2))
# DNN_model.add(Dense(units = 64, activation = swish))
# DNN_model.add(Dropout(0.2))
DNN_model.add(Dense(units = 1, activation = swish))
out1 = BatchNormalization()(out1)
out2 = BatchNormalization()(out2)
out3 = BatchNormalization()(out3)
out4 = BatchNormalization()(out4)
out5 = BatchNormalization()(out5)
out6 = BatchNormalization()(out6)

out1 = DNN_model(out1)
out2 = DNN_model(out2)
out3 = DNN_model(out3)
out4 = DNN_model(out4)
out5 = DNN_model(out5)
out6 = DNN_model(out6)

out = Activation('softmax')(concatenate([out1, out2, out3, out4, out5, out6]))

Final_model = Model(inputs = [input_context, input_target1, input_target2, input_target3, input_target4, input_target5, input_target6], 
                    outputs = out)
Final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


maxLen = 15
input_context = Input(shape = (45, ), dtype = 'int32')
input_target1 = Input(shape = (maxLen, ), dtype = 'int32')
input_target2 = Input(shape = (maxLen, ), dtype = 'int32')
input_target3 = Input(shape = (maxLen, ), dtype = 'int32')
input_target4 = Input(shape = (maxLen, ), dtype = 'int32')
input_target5 = Input(shape = (maxLen, ), dtype = 'int32')
input_target6 = Input(shape = (maxLen, ), dtype = 'int32')

tar_model = Sequential()
tar_model.add(embedding_layer2)
tar_model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True, kernel_initializer='he_uniform')))
tar_model.add(Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False, kernel_initializer='he_uniform')))
input_tar_embed1 = tar_model(input_target1)
input_tar_embed2 = tar_model(input_target2)
input_tar_embed3 = tar_model(input_target3)
input_tar_embed4 = tar_model(input_target4)
input_tar_embed5 = tar_model(input_target5)
input_tar_embed6 = tar_model(input_target6)

txt_model = Sequential()
txt_model.add(embedding_layer1)
txt_model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True, kernel_initializer='he_uniform')))
txt_model.add(Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False, kernel_initializer='he_uniform')))
input_ctx_embed = txt_model(input_context)

out1 = Dot(axes = 1, normalize = False)([input_tar_embed1, input_ctx_embed])
out2 = Dot(axes = 1, normalize = False)([input_tar_embed2, input_ctx_embed])
out3 = Dot(axes = 1, normalize = False)([input_tar_embed3, input_ctx_embed])
out4 = Dot(axes = 1, normalize = False)([input_tar_embed4, input_ctx_embed])
out5 = Dot(axes = 1, normalize = False)([input_tar_embed5, input_ctx_embed])
out6 = Dot(axes = 1, normalize = False)([input_tar_embed6, input_ctx_embed])

out = Activation('softmax')(concatenate([out1, out2, out3, out4, out5, out6]))

Final_model = Model(inputs = [input_context, input_target1, input_target2, input_target3, input_target4, input_target5, input_target6], 
                    outputs = out)
Final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


filep="NEWDIM200-{epoch:04d}-{val_acc:.5f}.h5"
checkpointer = ModelCheckpoint(monitor='val_acc', filepath=filep, verbose=1, save_best_only=True)
Final_model.fit_generator(data_generator(Q_bag2_seq, A_bag2_seq, Q_bag3_seq, A_bag3_seq, 128),steps_per_epoch=(len(Q_bag2_seq) + len(Q_bag3_seq))/128, validation_data = (val_data, val_L), nb_epoch=300, callbacks=[checkpointer])


# In[8]:


#model1 = load_model('weights-improvement-0002-0.62557.h5')
#model2 = load_model('weights-improvement-0004-0.79784.h5')


# In[9]:


testQ = doc2seq(re_sentences, word2index)
testA = doc2seq(next_sentences, word2index)


# In[10]:


predict_tmp1 = Final_model.predict([testQ ,testA])
predict_tmp = predict_tmp1 
#predict_tmp2 = model2.predict([testQ ,testA])
#predict_tmp = predict_tmp1 + predict_tmp2

length = int(len(predict_tmp1) / 6)
predict = predict_tmp.reshape((length, 6))

ans = np.zeros(length)
for i in range(length):
    ans[i] = int(predict[i].argmax())
    print(ans[i])


# In[11]:


text = open('final.csv', "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "ans"])
for i in range(len(ans)):
    s.writerow([str(i), int(ans[i])])
text.close()

