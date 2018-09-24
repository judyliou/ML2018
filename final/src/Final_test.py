# coding: utf-8
import gensim.models
import numpy as np
from gensim.models import word2vec
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Embedding, Dropout, Bidirectional, Input, BatchNormalization, GRU, Activation, merge, Dot, Add  
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence  
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from collections import defaultdict, Counter
from keras.layers import Activation
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import re
import csv

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

np.random.seed(5566)

def build_counter(texts):
    frequency = Counter()
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def build_w2i_map(fre_dic, threshold):
    MAX_FEATURES = threshold
    MAX_SENTENCE_LENGTH = 39

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

def bag4sentence(doc):
    bag = []
    ans = []
    for i in range(len(doc) - 4):
        bag.append(doc[i] + doc[i + 1] + doc[i + 2] + doc[i + 3])
        ans.append(doc[i + 4])
    return bag, ans


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

sentence = []
pick_num = 0
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


txt_counter = build_counter(sentence)
word2index, index2word = build_w2i_map(txt_counter, len(txt_counter))


model1 = load_model('models/BIAS300_best.h5')
model2 = load_model('models/Double_LSTM.h5')
model3 = load_model('models/DIM100-0016-0.64692.h5')


testA = doc2seq(next_sentences, word2index, 15)
A_length = testA.shape[0]
testA = testA.reshape(int(A_length / 6), 6 * 15)
testQ = doc2seq(pre_sentences, word2index, 45)
test_data = [testQ] + np.hsplit(testA, 6)
predict1 = model1.predict(test_data)
print('done1')
predict2 = model2.predict(test_data)
print('done2')
predict3 = model3.predict(test_data)
print('done3')

model4 = load_model('models/single_best.h5')
model5 = load_model('models/double_best.h5')
print('load done')
testA = doc2seq(next_sentences, word2index, 45)
print(testA.shape)
A_length = testA.shape[0]
testA = testA.reshape(int(A_length / 6), 6 * 45)
testQ = doc2seq(pre_sentences, word2index, 45)
test_data = [testQ] + np.hsplit(testA, 6)
predict4 = model4.predict(test_data)
print('done4')
predict5 = model5.predict(test_data)
print('done5')

predict = predict1 + predict2 + predict3 + predict4 + predict5

ans = np.zeros(len(predict))
for i in range(len(predict)):
    ans[i] = int(predict[i].argmax())
    print(ans[i])

f=open("data/testing_data.csv",'r',encoding='utf-8')
f.readline()
count=0
for line in f:
    a=line.split(",")
    q=a[1].lstrip().rstrip().split()
    b=re.split("[0-5] *:",a[2])[1:7]
    x=7
    for item in q:
        x=7
        if txt_counter[item]<1000:
            for c in range(len(b)):
                if item in b[c]:
                    if x==7:
                        x=c
                    elif x!=7:
                        x=9
            if x!=7 and x!=9:
                ans[count]=x
                break
    count+=1
f.close()

text = open('final.csv', "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "ans"])
for i in range(len(ans)):
    s.writerow([str(i), int(ans[i])])
text.close()

