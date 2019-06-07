import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D

from getkeys import key_check
import random

import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


WIDTH = 300
HEIGHT = 169

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        # model = load_model('DEEP_DRIVE_5_29_19_19_1559270337.573312.h5') 
        model = load_model('DEEP_DRIVE_5_26_19_3_1558958288.0424793.h5')
        # model = load_model('DEEP_DRIVE_5_26_19_1.model')
# adagrad mean_squared_error
# model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adagrad', loss = 'mean_squared_error', metrics=['accuracy'])
def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,40,1280,760))
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (300,169))
            screen = screen.reshape(1, 300, 169, 1)
            prediction = model.predict(screen)[0]
            # prediction = np.array(prediction) - np.array([0.2, 1.0, 1.0, 1.0,  1.0,   1.0, 1.0, 1.0, 1.0])
            print(prediction)

            if np.argmax(prediction) == np.argmax(w):
                straight()
                # time.sleep(0.11)
            
            elif np.argmax(prediction) == np.argmax(s):
                # reverse()
                straight()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(a):
                left()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(d):
                right()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(wa):
                left()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(wd):
                right()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(sa):
                forward_right()
                # reverse_left()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(sd):
                forward_left()
                # reverse_right()
                time.sleep(0.11)
            if np.argmax(prediction) == np.argmax(nk):
                straight()
                time.sleep(0.11)
            
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
