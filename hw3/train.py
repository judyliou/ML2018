# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:37:18 2018

@author: Owner
"""

import sys
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
        
class_num = 7
train_size = 28000
batch = 128
epoch = 50
model_name = 'model.h5'
output_name = 'predict.csv'
  

x, y = [], []
with open(sys.argv[1], "r") as f:
    n_row = 0
    for r in list(csv.reader(f))[1:]:
        y.append(int(r[0]))
        x.append([float(i) for i in r[1].split()])
        n_row += 1
        if n_row % 5000 == 0:
            print('row:' + str(n_row))

#preprocessing
x_train = np.array(x)
x_train /= 255
mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train - mean) / std
y_train = to_categorical(y, class_num)

#validation set
data = np.concatenate((y_train, x_train), axis = 1)
np.random.shuffle(data)
x_train, y_train = data[:train_size, class_num:], data[:train_size, :class_num]
x_val, y_val = data[train_size:, class_num:], data[train_size:, :class_num]

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)


#Build Model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape = (48, 48,1)))
model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.02))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.02))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.04))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.04))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.1))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(class_num))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1)

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    zoom_range = 0.2)
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch),
                    steps_per_epoch = len(x_train) // batch, 
                    epochs=epoch)

result = model.evaluate(x_train, y_train)
print('Train Acc:', result[1])

model.save(model_name)
print('Save:', model_name)

val_result = model.predict(x_val, batch_size = batch, verbose = 1)
acc = 0
for i in range(len(val_result)):
    val_class = np.argmax(val_result[i])
    if y_val[i][val_class] == 1:
        acc += 1
print('Validation Acc: %.4f' % (acc/len(x_val)))
