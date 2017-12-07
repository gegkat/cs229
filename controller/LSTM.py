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

import models
import argparse
import matplotlib
matplotlib.use("Agg")#for cluster which doesn't have a display
from matplotlib import pyplot
import pickle
from datetime import datetime
import json

import sys

#TO RUN:
##python <filename> <architecture_type> <fraction_of_examples_to_test>
##python train_models.py NVIDIA 0.3

def split_train(lis,n):
        num_sp=int(len(lis)*n)
        return lis[:num_sp]


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def mkdir_unique(timestamp, args):
    prefix='time_'+str(args.timesteps)+'_epochs_'+str(args.epochs)
    mydir = os.path.join(
        os.getcwd(), 
        'LSTM'+ '_' + prefix+ '_'+timestamp)

    os.makedirs(mydir)

    return mydir





def get_samples(data_dir, num_split, timesteps):

    # Hard-coded constants
    STEERING_CORRECTION = [0, 0.2, -0.2] # Steering correction for center, left, right images

    # Large turns are the biggest challenge for the model, but the majority of the samples
    # represent driving straight. The following constants are used to discard a portion of
    # of samples under a minimum steering angle
    SMALL_TURN_THRESH = 0.03  # Threshold to consider discarding a sample
    SMALL_TURN_DISCARD_PROBABILITY = 0.60   # Probability to discard a sample

    # Read and store all lines in .csv file
    data_dir = './' + data_dir + '/'
    print('Start read csv')
    start_time = time.time()
    lines = []
    with open(data_dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    end_time = time.time()
    print('Read {} lines if csv in {:.2f} seconds'.format(len(lines), end_time-start_time))

#    lines = sklearn.utils.shuffle(lines)
    # Preprocess the lines of the csv file
    # At the end of this step we will have a list of samples. Each sample
    # is a list with three eleements: 
    #   1. File name to an image file, including right and left images
    #   2. A steering angle (with correction factor for left/right images)
    #   3. A flag to flip or not flip the image
    # There will be 6 times the number of samples as there were lines in the
    # csv file minus some samples being removed for small steering angles
    lines_split= split_train(lines,num_split)
    images_center = []
    images_right=[]
    images_left=[]
    for line in lines_split:
        file_name = data_dir + 'IMG/' + line[0].split('/')[-1]
        images_center.append(file_name)
        file_name = data_dir + 'IMG/' + line[1].split('/')[-1]
        images_left.append(file_name)
        file_name = data_dir + 'IMG/' + line[2].split('/')[-1]
        images_right.append(file_name)

    samples=[]
    for i in range(0,len(lines)-timesteps):
        orig_steering_angle = float(lines_split[i+timesteps-1][3])
        orig_throttle = float(lines_split[i+timesteps-1][4])
        orig_brake = float(lines_split[i+timesteps-1][5])
        orig_speed = float(lines_split[i+timesteps-1][6])
        if orig_steering_angle<SMALL_TURN_THRESH and random.random()<=SMALL_TURN_DISCARD_PROBABILITY:
            continue
        samples.append([images_center[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[0],orig_throttle,orig_speed,True])
        samples.append([images_center[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[0],orig_throttle,orig_speed,False])
        samples.append([images_left[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[1],orig_throttle,orig_speed,True])
        samples.append([images_left[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[1],orig_throttle,orig_speed,False])
        samples.append([images_right[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[2],orig_throttle,orig_speed,True])
        samples.append([images_right[i:i+timesteps],orig_steering_angle+STEERING_CORRECTION[2],orig_throttle,orig_speed,False])
    print('# SAMPLES: {}'.format(len(samples)))
    return samples



# Define generator function for use by keras fit_generator function
def generator(samples, dimensions,output_throttle=False,batch_size=32):
    num_samples = len(samples)
    timesteps,rows,cols,ch=dimensions
    while 1: # Loop forever so the generator never terminates
        # Shuffle for each loop through the data
        samples = sklearn.utils.shuffle(samples)
        # Loop through data in batches
        for offset in range(0, num_samples, batch_size):
            # Get a batch of samples
            batch_samples = samples[offset:offset+batch_size]

            # For each sample read the img data from file
            images = []
            angles = []
            throttles = []
            speeds = []
            for batch_sample in batch_samples:
                do_flip_flag = batch_sample[4]
                image=np.zeros([timesteps,rows,cols,ch])
                for i in range(0,timesteps):
                    read_image=cv2.imread(batch_sample[0][i],0)
                    if do_flip_flag:
                        read_image = cv2.flip(read_image, 1)
                    image[i,:,:,:] =np.reshape(read_image,[rows,cols,ch])
                angle=batch_sample[1]
                if do_flip_flag:
                    angle = -angle
                throttle = batch_sample[2]
                speed = batch_sample[3]
                images.append(image)
                angles.append(angle)
                throttles.append(throttle)
            # Convert to numpy array for Keras
            X_train = np.array(images)
            if output_throttle:
                y_train = np.column_stack((angles, throttles))
            else:
                y_train = np.array(angles)
            yield X_train, y_train


# Constants
ch, row, col = 1, 160, 320  # Trimmed image format

# Include throttle in output of generator
OUTPUT_THROTTLE = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument(
        'data_dir',
        type=str,
        help='Relative path to directory containing training data'
    )


    parser.add_argument(
        '--frac',
        type=float,
        default = 1.0,
        help='Fraction of example data to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default = 3,
        help='# of Epochs to train'
    )

    parser.add_argument(
        '--showplot',
        type=int,
        default = 0,
        help='Flag to call pyplot.show(). 1 to show plot otherwise will not show'
    )

    parser.add_argument(
        '--batchsize',
        type=int,
        default = 64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default = 10,
        help='Time steps to consider per example'
                        )

    args = parser.parse_args()
    samples = get_samples(args.data_dir, args.frac,args.timesteps)
    # Split samples into training and validation
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    timesteps=args.timesteps
    # compile and train the model using the generator function
    train_generator = generator(train_samples,(timesteps,row,col,ch), output_throttle=OUTPUT_THROTTLE,batch_size=args.batchsize)
    validation_generator = generator(validation_samples,(timesteps,row,col,ch), output_throttle=OUTPUT_THROTTLE,batch_size=args.batchsize)

    if OUTPUT_THROTTLE:
        n_outputs = 2
    else:
        n_outputs = 1

	model=Sequential()
	model=models.LSTM_model(model,n_outputs,(timesteps,ch,row,col),0)
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    # Train model
    start_time = time.time()
    train_steps_per_epoch = floor(len(train_samples)/args.batchsize)
    val_steps_per_epoch = floor(len(validation_samples)/args.batchsize)
    if val_steps_per_epoch == 0:
        val_steps_per_epoch = 1
    history = model.fit_generator(train_generator, steps_per_epoch= train_steps_per_epoch
                , validation_data=validation_generator, 
                validation_steps= val_steps_per_epoch, epochs=args.epochs)
    end_time = time.time()
    print('Trained model in {:.2f} seconds'.format(end_time-start_time))

    timestamp = get_timestamp()
    udir = mkdir_unique(timestamp, args)
    # Save the model
    print("Saving model weights and configuration file.")
    model.save(os.path.join(udir,'model_LSTM_' + str(args.timesteps)+'.h5'))

    with open(os.path.join(udir, 'history.csv'), 'w') as f:
        for i in range(0, len(history.history['loss'])):
            f.write("{}, {}\n".format(history.history['loss'][i], history.history['val_loss'][i]))

    with open(os.path.join(udir, 'train_history_dict.pickle'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    with open(os.path.join(udir, 'config.log'), 'w') as f:
        f.write(json.dumps(vars(args)))
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    with open(os.path.join(udir, 'model.json'), 'w') as f:
        f.write(json.dumps(model.to_json()))
    pyplot.plot(history.history['loss'], label='training loss')
    pyplot.plot(history.history['val_loss'], label='val loss')
    pyplot.legend()
    pyplot.xlabel('# Epochs')
    pyplot.ylabel('MSE Loss')
    pyplot.savefig(os.path.join(udir, 'loss_vs_epoch.png'), dpi=400)
    if args.showplot == 1:
        pyplot.show()

