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

def LSTM_model(model,no_outputs,dimensions, verbosity=0,dropout=0):
    timesteps, ch, row, col = dimensions
    model = Sequential()
    if timesteps == 1:
        raise ValueError("Not supported w/ TimeDistributed layers")
  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer
    model.add(Lambda(lambda x: x/127.5 - 1.,
      input_shape=(timesteps,  row, col,ch),
      )
  )

    model.add(TimeDistributed(Cropping2D(cropping=((70,25),(0,0)))))

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ReLU activation
    model.add(TimeDistributed(
      Convolution2D(24, 5, strides=(2, 2), activation='relu')))
    model.add(TimeDistributed(
      Convolution2D(36, 5, strides=(2, 2), activation='relu')))
    model.add(TimeDistributed(
      Convolution2D(48, 5, strides=(2, 2), activation='relu')))
    model.add(TimeDistributed(
      Convolution2D(64, 3, activation='relu')))
    model.add(TimeDistributed(
      Convolution2D(64, 3, activation='relu')))


    model.add(TimeDistributed(Flatten()))
# Takes in sequences and returns entire predicted sequences
#    model.add(LSTM(512,return_sequences=True))
#Does not return sequences, just takes final output
    model.add(LSTM(100))
#Fully connected layers on final output
    model.add(Dense(100))
    if(dropout!=0):
        model.add(Dropout(dropout))
    model.add(Dense(50,activation='relu'))
    if(dropout!=0):
        model.add(Dropout(dropout))
    model.add(Dense(10,activation='relu'))
#Predict only steering for now
    model.add(Dense(no_outputs))
  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
    model.compile(optimizer="adam", loss="mse")
    if verbosity > 1:
        config = model.get_config()
        for layerSpecs in config:
            pprint(layerSpecs)
    return model

def NVIDIA(model, n_outputs=1):

    # NVIDIA architechture
    model.add(Convolution2D(24,5, strides=(2,2), activation='relu'))
    model.add(Convolution2D(36,5, strides=(2,2), activation='relu'))
    model.add(Convolution2D(48,5, strides=(2,2), activation='relu'))
    model.add(Convolution2D(64,3, activation='relu'))
    model.add(Convolution2D(64,3, activation='relu'))
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(n_outputs))


    return model

def complex_CNN(model, n_outputs=1):
    model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(40,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(50,(5,5),strides=(2,2),activation="relu"))
    model.add(Flatten())
    model.add(Dense(100,activation="tanh"))
    model.add(Dense(50,activation="tanh"))
    model.add(Dense(10,activation="tanh"))
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
