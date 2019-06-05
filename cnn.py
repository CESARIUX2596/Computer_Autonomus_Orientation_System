import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import pickle
import random
import pandas as pd
import time

def a_net(WIDTH, HEIGHT):
  model = Sequential()
  
  model.add(Conv2D(96, (11,11), strides=4, input_shape=(WIDTH, HEIGHT,1)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))
  
  model.add(Conv2D(256, (5,5)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))
  
  model.add(Conv2D(384, (3,3)))
  model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(3,3), strides=2))
  model.add(Conv2D(384, (3,3)))
  model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(3,3), strides=2))
  model.add(Conv2D(256, (3,3)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(3,3), strides=2))

  model.add(Conv2D(256, (5,5)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))

  model.add(Conv2D(384, (3,3)))
  model.add(Activation('relu'))
  model.add(Conv2D(384, (3,3)))
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))
  
  model.add(Flatten())
  model.add(Dense(4096))
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  
  model.add(Dense(4096))
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  
  model.add(Dense(4096))
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))

  model.add(Dense(4096))
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  
  model.add(Dense(9))
  model.add(Activation('softmax'))
  
  return model
  # Compile the model


WIDTH = 300
HEIGHT = 169
time = time.time()
MODEL_NAME = "DEEP_DRIVE_5_26_19_2_{}.h5".format(time)
sgd = keras.optimizers.SGD(lr=0.01, clipnorm=1.0)
#model = model1(WIDTH, HEIGHT)

#model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model  = a_net(WIDTH,HEIGHT)
model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sdg', metrics=["accuracy"])

EPOCHS = 15

order = np.linspace(1,75,75)
random.seed( 303030 )
random.shuffle(order)
count = 0
for e in range(EPOCHS):
  for o in range(len(order)):
    print(e,o)
    train_data = np.load('D:/MODEL/Deep Drive/MODEL/TRAIN_DATA/deepdrive_data-{}.npy'.format(int(order[o])))    
    print('training_data-{0}.npy'.format(int(order[o])), len(train_data))    
    train = train_data[:-50]
    test = train_data[-50:]
    
    X = np.asarray([i[0] for i in train])
#     print(X.shape)
    print(X.shape)
#     train_X = []
#     X = X.astype('float32')
    train_X = np.asarray(X).reshape(-1,300,169,1)
    train_X = train_X.astype('float32')
    train_X /= 255.0
    train_X = np.array(train_X)
#     print(train_X.shape)
    Y = [i[1] for i in train]
    test_x = np.array([i[0] for i in test])
#     test_x = test_x.reshape(-1,160,120,1)
#     test_x = test_x.astype('float32')
    test_y = [i[1] for i in test]
    
    Y = np.array(Y)
#     X /= 255.0
#     test_x /= 255.0
    
    model.fit(train_X,Y, epochs = 1)
    
    model.save(MODEL_NAME)

print('DONE!!!!!')