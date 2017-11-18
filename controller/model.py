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

# Constants
ch, row, col = 3, 160, 320  # Trimmed image format
BATCH_SIZE = 64 # Used in generator to load images in batches
MAX_SAMPLES = 50000 # Used for testing to reduce # of files to use in training
STEERING_CORRECTION = [0, 0.2, -0.2] # Steering correction for center, left, right images
DIR = './2017_11_12 training data/' # Directory for driving log and images

# Large turns are the biggest challenge for the model, but the majority of the samples
# represent driving straight. The following constants are used to discard a portion of
# of samples under a minimum steering angle
SMALL_TURN_THRESH = 0.03  # Threshold to consider discarding a sample
SMALL_TURN_PROBABILITY = 0.60   # Probability to discard a sample

# Read and store all lines in .csv file
print('Start read csv')
start_time = time.time()
lines = []
with open(DIR + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
end_time = time.time()
print('Read {} lines if csv in {:.2f} seconds'.format(len(lines), end_time-start_time))

# Preprocess the lines of the csv file
# At the end of this step we will have a list of samples. Each sample
# is a list with three eleements: 
#   1. File name to an image file, including right and left images
#   2. A steering angle (with correction factor for left/right images)
#   3. A flag to flip or not flip the image
# There will be 6 times the number of samples as there were lines in the
# csv file minus some samples being removed for small steering angles
samples = []
for line in lines: 
    # Initialize to true
    use_sample = True

    # Pull the steering angle from the 4th column
    orig_steering_angle = float(line[3]) 
    orig_throttle = float(line[4]) 
    orig_brake = float(line[5]) 
    orig_speed = float(line[6])

    # Skip over a portion of the samples with small steering angles
    if np.abs(orig_steering_angle) < SMALL_TURN_THRESH:
        if random.uniform(0,1) <= SMALL_TURN_PROBABILITY:
            use_sample = False

    if use_sample:
        # Loop through center, left, and right images
        for i in range(0,3):
            # Pull the file name from the log
            file_name = DIR + 'IMG/' + line[i].split('/')[-1]

            # Add a correction factor for left and right cameras
            steering_angle = orig_steering_angle + STEERING_CORRECTION[i]

            # Add two copies of each image. One for regular and one for reversing the image
            # The reversing is done after the image is read in the generator. Here we are
            # just setting the flag true or false to tell the generator whether to reverse
            samples.append([file_name, steering_angle, orig_throttle, orig_speed, False])
            samples.append([file_name, steering_angle, orig_throttle, orig_speed, True])

# Shuffle the samples
samples = sklearn.utils.shuffle(samples)

# For testing purposes, limit the number of samples if desired
if len(samples) > MAX_SAMPLES:
    samples = samples[0:MAX_SAMPLES]

print('# SAMPLES: {}, {}'.format(len(samples)/6, len(samples)))

# Split samples into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define generator function for use by keras fit_generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
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
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]
                throttle = batch_sample[2]
                speed = batch_sample[3]
                do_flip_flag = batch_samples[4]

                # Half of the samples are have a boolean for flipping the image
                if do_flip_flag:
                    image = cv2.flip(image, 1)
                    angle = -1.0*angle
                images.append(image)
                angles.append(angle)
                throttles.append(throttle)

            # Convert to numpy array for Keras
            X_train = np.array(images)
            y_train = np.column_stack((angles, throttles))
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# Test generator
count = 0
for X,y in generator(train_samples, BATCH_SIZE):
    count = count+1
    if count >= 2:
        break
print('Generator test: X.shape = {}, y.shape = {}'.format(X.shape, y.shape))
#print(y[0:5])

# Define the neural net
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# Crop to focus on the part of the image containing the road
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')

# Train model
start_time = time.time()
model.fit_generator(train_generator, steps_per_epoch= 
            len(train_samples)/BATCH_SIZE-1, validation_data=validation_generator, 
            validation_steps=floor(len(validation_samples)/BATCH_SIZE), epochs=3)
end_time = time.time()
print('Trained model in {:.2f} seconds'.format(end_time-start_time))

# Save the model
print("Saving model weights and configuration file.")
model.save('model.h5')
