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

sio = socketio.Server()
app = Flask(__name__)
model = None

global img_list


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


@sio.on('telemetry')
def telemetry(sid, data):
    global img_list

    if data:

        with open(out_file, 'a') as f: 
            keys = ['steering_angle', 'throttle', 'speed', 'x', 'z', 'heading']
            for key in keys:
                f.write('{}, '.format(data[key]))
            f.write('\n')

        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
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
        print("--- %s seconds ---" % (time.time() - start_time))

        # Pull out steering angle and throttle (optional) from 
        # model output
        model_output = model_output[0]
        steering_angle = float(model_output[0])

        # Use throttle model if given, otherwise use PID controller
        if len(model_output) > 1:
            throttle = float(model_output[1])
        else:
            throttle = controller.update(float(speed))

        # Send controls to simulator
        send_control(steering_angle, throttle)

        # save frame for video
        if args.img_dir != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.img_dir, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

#    simulator=sp.Popen('../carsim_mac.app/Contents/MacOS/carsim_mac')
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--record',
        type=int,
        nargs='?',
        default=0,
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    parser.add_argument(
        '--speed',
        type=int,
        default=12,
        help='Set desired speed for PID controller'
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

    args = parser.parse_args()

    # Initialize global img_list
    global img_list
    img_list = []

    # Open telem file
    out_file = args.model + '_' + str(args.speed) + 'mph.telem'
    with open(out_file, 'w') as f:
        # just clear contents of out_file
        pass

    # Default img_dir is empty, but if record flag is true then set the dir
    args.img_dir = ''
    if args.record == 1:
        args.img_dir = args.model + '_img'

    # Initialize controller gains and desired speed
    controller = SimplePIController(0.1, 0.002)
    controller.set_desired(args.speed)

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.img_dir != '':
        print("Creating image folder at {}".format(args.img_dir))
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)
        else:
            shutil.rmtree(args.img_dir)
            os.makedirs(args.img_dir)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
