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

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def xtrack(x,z,x_ref,z_ref):
	#finding the nearest waypoint for each given x,z
	near=np.zeros(np.size(x_ref))
	for i in range(0, np.size(x_ref)):
		near[i]= np.square(float(x)-x_ref[i])+ np.square(float(z)-z_ref[i])
	arg=np.argmin(near)
	#base_x=x_ref[:arg] 	
	#base_z=z_ref[:arg]
	#s=0 
	#for k in range(0,arg-1):
	#	s=s+(x_ref[k+1]-x_ref[k])**2+ (z_ref[k+1]-z_ref[k])**2	
	if(arg==(np.size(x_ref)-1)):
		fwarg=0
	else:
		fwarg=arg+1
	#print("fwarg="+str(fwarg))
	vec1=np.array([x-x_ref[arg], z-z_ref[arg]])
	vec2=np.array([x_ref[fwarg]-x_ref[arg],z_ref[fwarg]-z_ref[arg]])
	l=int(0)
	if(np.cross(vec2,vec1)<0):
		l=1
	#dp=np.dot(vec1,vec2)
	if(np.dot(vec1,vec2)>0):
		#narg=arg+1
		narg=fwarg
		#print("next")
	else:
		narg=arg-1
		#print("previous")
	#print("arg:"+str(arg)+"nextarg:"+str (narg))
	#print("x:"+str(x)+"z:"+str (z))
	#print("point arg:"+str(x_ref[arg])+","+str(z_ref[arg])+"point nextarg:"+str (x_ref[narg])+","+str(z_ref[narg]))
	h=(x-x_ref[arg])**2+(z-z_ref[arg])**2
	v1=np.array([x-x_ref[arg],z-z_ref[arg]])
	v2=np.array([x_ref[narg]-x_ref[arg],z_ref[narg]-z_ref[arg]])
	b=((np.dot(v1,v2))/(np.linalg.norm(v2)))**2
	#num=(abs((z_ref[narg]-z_ref[arg])*(x)-(x_ref[narg]-x_ref[arg])*(z)+ (x_ref[narg]*z_ref[arg]-z_ref[narg]*x_ref[arg])))**2
	#den=((z_ref[narg]-z_ref[arg])**2+(x_ref[narg]-x_ref[arg])**2)
	#d_sq=(num/den)**(0.5)
	d_sq=(h-b)**(0.5)
	#print("dist from center:"+str(d_sq))
	return(d_sq,l)


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
    f1= 'lake_track_waypoints.csv'
    with open(f1) as f:
    	content= f.readlines()
    	content=[m.strip() for m in content]
	#print(content)
	#getting x and z from the lines
    	x=[i.split(',',1)[0] for i in content]
    	z=[i.split(',',1)[1] for i in content]
    f.close()

#np array creation
    x_ref=np.asarray(x,dtype=float)
    z_ref=np.asarray(z,dtype=float)

    if data:
        #print('start')
        #for key in data:
        #    print(key)

        with open(out_file, 'a') as f: 
            keys = ['steering_angle', 'throttle', 'speed', 'x', 'z', 'heading']
            for key in keys:
                f.write('{}, '.format(data[key]))
            f.write('\n')

        #print('end')
        car_x=data[key].split(',',6)[3]
        car_z=data[key].split(',',6)[4]
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
        model_output = model.predict(image_array[None, :, :, :], batch_size=1) 
        model_output = model_output[0]
	#ADDITION, this is for the case of changing the steering when the car is at extreme end of the road
	cte,l=xtrack(car_x,car_z,x_ref,z_ref)
	if(cte>=2.75 and cte<=3 and l==1):
        	steering_angle = float(3.00)
	elif(cte>=2.75 and cte<=3 and l==0):
        	steering_angle = float(-3.00)
	else:
        	steering_angle = float(model_output[0])
 
        #steering_angle = float(model_output[0])
        if len(model_output) > 1:
            throttle = float(model_output[0][1])
        else:
            throttle = controller.update(float(speed))

        #print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
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
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    out_file = args.model + '_' + str(args.speed) + 'mph.telem'
    with open(out_file, 'w') as f:
        # just clear contents of out_file
        pass

    # Default img_dir is empty, but if record flag is true then set the dir
    args.img_dir = ''
    if args.record == 1:
        args.img_dir = args.model + '_img'


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
