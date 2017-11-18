from math import *
import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import time
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random


from math import *
import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, TimeDistributed, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
# from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.models import Sequential
from keras.layers import LSTM
import time
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random

def NVIDIA(model, n_outputs=1):

    # NVIDIA architechture
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(n_outputs))


    return model


def linear_regression(model, n_outputs=1):

    # NVIDIA architechture
    model.add(Flatten())
    model.add(Dense(1,activation='linear'))

    return model

def simple_NN(model, n_outputs=1):

    # NVIDIA architechture
    model.add(Flatten())

    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))


    return model


def simple_CNN(model, n_outputs=1):

    # NVIDIA architechture
    model.add(Convolution2D(10,5,5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_outputs))


    return model
