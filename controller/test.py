import argparse
import base64
from datetime import datetime
import os
import shutil
import sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import subprocess as sp
import time

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

import cv2

import csv


sio = socketio.Server()
app = Flask(__name__)
model = None

global img_list


def predict(image):
    global img_list

    image_array = np.asarray(image)

    # Begin timer
    start_time = time.time()

    # Convert image to gray
    if args.gray:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        b,m=np.shape(image_array)
        image_array = np.reshape(image_array, (b,m,1))

    if args.LSTM > 0:
        img_list.append(image_array)
        if len(img_list) == args.LSTM:
            model_in = np.array(img_list)
            model_output = model.predict(model_in[None, :, :, :, :], batch_size=1) 

            # pop off first img in list
            img_list.pop(0)
        else: 
            # If we don't have enough images return 0 for controls
            model_output = [[0, 0]]


    else:
        model_output = model.predict(image_array[None, :, :, :], batch_size=1) 

    # time.sleep(0.02)
    # Report time for model to predict
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Pull out steering angle and throttle (optional) from 
    # model output
    model_output = model_output[0]
    steering_angle = float(model_output[0])

    # Use throttle model if given, otherwise use PID controller
    if len(model_output) > 1:
        throttle = float(model_output[1])
    else:
        throttle = 0 

    return (steering_angle, throttle)



if __name__ == '__main__':

#    simulator=sp.Popen('../carsim_mac.app/Contents/MacOS/carsim_mac')
    parser = argparse.ArgumentParser(description='Remote Driving')

    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    parser.add_argument(
        '--gray',
        type=int,
        default=0,
        help='Boolean to control whether to convert input images to grayscale.'
    )

    parser.add_argument(
        '--LSTM',
        type=int,
        default=0,
        help='# of images to pass as array to LSTM. Use 0 if not LSTM.'
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='./',
        help='Path to directory containing training data'
    )

    args = parser.parse_args()

    # Initialize global img_list
    global img_list
    img_list = []


    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    data_dir = args.train_dir
    lines = []
    with open(data_dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    STEERING_CORRECTION = [0, 0.2, -0.2] # Steering correction for center, left, right images

    out_file = args.model + '.test'
    out_files = [ out_file + '.center', out_file + '.left', out_file + '.right',]
    # Loop through center, left, and right images
    for i in range(0,3):
        count = 0
        with open(out_files[i], 'w') as f:
            for line in lines: 
                print("{} {}".format(i, len(lines) - count))
                count += 1

                # Pull the steering angle from the 4th column
                orig_steering_angle = float(line[3]) 
                orig_throttle = float(line[4]) 
                orig_brake = float(line[5]) 
                orig_speed = float(line[6])

                # Pull the file name from the log
                file_name = data_dir + 'IMG/' + line[i].split('/')[-1]

                # Add a correction factor for left and right cameras
                steering_angle = orig_steering_angle + STEERING_CORRECTION[i]

                image = cv2.imread(file_name)

                (steering, throttle) =  predict(image)
                f.write('{}, {}, {}, {}, {}, {} \n'.format(
                    file_name, steering_angle, steering, 
                    orig_throttle, orig_brake, throttle))
