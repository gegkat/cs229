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

def buildNVIDIAModell():
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

    return model


def buildLSTMModel(timesteps, cameraFormat, verbosity=0):
  """
  Build and return a CNN + LSTM model; details in the comments.
  The model expects batch_input_shape =
  (volumes per batch, timesteps per volume, (camera format 3-tuple))
  A "volume" is a video frame data struct extended in the time dimension.
  Args:
    volumesPerBatch: (int) batch size / timesteps
    timesteps: (int) Number of timesteps per volume.
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, height, width).
    verbosity: (int) Print model config.
  Returns:
    A compiled Keras model.
  """
  print("Building model...")
  ch, row, col = cameraFormat

  model = Sequential()

  if timesteps == 1:
    raise ValueError("Not supported w/ TimeDistributed layers")

  print(timesteps, ch, row, col)

  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer in order to
  # have stateful recurrent layers later
  model.add(Lambda(
      lambda x: x/127.5 - 1.,
      input_shape=(timesteps, ch, row, col),
      )
  )

  model.add(TimeDistributed(Cropping2D(cropping=((70,25),(0,0)))))

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ELU activation
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

  model.add(LSTM(100,
                 return_sequences=True))
  # model.add(LSTM(512,
  #                return_sequences=True,
  #                stateful=True))


  model.add(TimeDistributed(Dense(100)))
  model.add(TimeDistributed(Dense(50)))
  model.add(TimeDistributed(Dense(10)))
  model.add(TimeDistributed(Dense(2)))

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  model.compile(optimizer="adam", loss="mse")
  return model


def buildLSTMModel_fast(timesteps, cameraFormat, verbosity=0):
  """
  Build and return a CNN + LSTM model; details in the comments.
  The model expects batch_input_shape =
  (volumes per batch, timesteps per volume, (camera format 3-tuple))
  A "volume" is a video frame data struct extended in the time dimension.
  Args:
    volumesPerBatch: (int) batch size / timesteps
    timesteps: (int) Number of timesteps per volume.
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, height, width).
    verbosity: (int) Print model config.
  Returns:
    A compiled Keras model.
  """
  print("Building model...")
  ch, row, col = cameraFormat

  model = Sequential()

  if timesteps == 1:
    raise ValueError("Not supported w/ TimeDistributed layers")

  print(timesteps, ch, row, col)

  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer in order to
  # have stateful recurrent layers later
  model.add(Lambda(
      lambda x: x/127.5 - 1.,
      input_shape=(timesteps, ch, row, col),
      )
  )

  model.add(TimeDistributed(Cropping2D(cropping=((70,25),(0,0)))))

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ELU activation
  model.add(TimeDistributed(
      Convolution2D(12, 5, strides=(2, 2), activation='relu')))
  model.add(TimeDistributed(
      Convolution2D(18, 5, strides=(2, 2), activation='relu')))
  model.add(TimeDistributed(
      Convolution2D(32, 3, activation='relu')))


  model.add(TimeDistributed(Flatten()))

  model.add(LSTM(50,
                 return_sequences=True))
  # model.add(LSTM(512,
  #                return_sequences=True,
  #                stateful=True))


  model.add(TimeDistributed(Dense(50)))
  model.add(TimeDistributed(Dense(10)))
  model.add(TimeDistributed(Dense(2)))

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  model.compile(optimizer="adam", loss="mse")

  # if verbosity:
  #   printTemplate = PrettyTable(["Layer", "Input Shape", "Output Shape"])
  #   printTemplate.align = "l"
  #   printTemplate.header_style = "upper"
  #   for layer in model.layers:
  #     printTemplate.add_row([layer.name, layer.input_shape, layer.output_shape])
  #   print(printTemplate)

  if verbosity > 1:
    config = model.get_config()
    for layerSpecs in config:
      pprint(layerSpecs)
    
  return model

# Constants
ch, row, col = 3, 160, 320  # Trimmed image format
BATCH_SIZE = 200 # Used in generator to load images in batches
TIME_STEPS = 5
MAX_SAMPLES = 50000 # Used for testing to reduce # of files to use in training
STEERING_CORRECTION = [0, 0.2, -0.2] # Steering correction for center, left, right images
DIR = './2017_11_12 training data/' # Directory for driving log and images
# DIR = './2017_11_17_slow/'
# Large turns are the biggest challenge for the model, but the majority of the samples
# represent driving straight. The following constants are used to discard a portion of
# of samples under a minimum steering angle
SMALL_TURN_THRESH = 0 #0.03  # Threshold to consider discarding a sample
SMALL_TURN_PROBABILITY = 0 #0.60   # Probability to discard a sample

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
        # for i in range(0,3):
        for i in range(0,1):
            # Pull the file name from the log
            file_name = DIR + 'IMG/' + line[i].split('/')[-1]

            # Add a correction factor for left and right cameras
            steering_angle = orig_steering_angle + STEERING_CORRECTION[i]

            # Add two copies of each image. One for regular and one for reversing the image
            # The reversing is done after the image is read in the generator. Here we are
            # just setting the flag true or false to tell the generator whether to reverse
            samples.append([file_name, steering_angle,  orig_throttle, orig_brake, orig_speed, False])
            #samples.append([file_name, steering_angle,  orig_throttle, orig_brake, orig_speed, True])

# Shuffle the samples
# samples = sklearn.utils.shuffle(samples)

# For testing purposes, limit the number of samples if desired
if len(samples) > MAX_SAMPLES:
    samples = samples[0:MAX_SAMPLES]

# print('# SAMPLES: {}, {}'.format(len(samples)/6, len(samples)))
print('# SAMPLES: {}'.format(len(samples)))

# Split samples into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define generator function for use by keras fit_generator function
def generator(samples, batch_size=120, time_steps=10):
    num_samples = len(samples)
    volumes_per_batch = batch_size / time_steps
    while 1: # Loop forever so the generator never terminates
        # Shuffle for each loop through the data
        # samples = sklearn.utils.shuffle(samples)

        # Loop through data in batches
        for offset in range(0, num_samples, batch_size):

            # Get a batch of samples
            batch_samples = samples[offset:offset+batch_size]

            X_volumes = []
            y_volumes = []
            for offset2 in range(0, len(batch_samples), time_steps):

                # Get a volume of images
                volume_samples = batch_samples[offset2:offset2+time_steps]


                # For each sample read the img data from file
                images = []
                angles = []
                throttles = []
                speeds = []
                for sample in volume_samples:
                    image = cv2.imread(sample[0])
                    angle = sample[1]
                    throttle = sample[2]
                    speed = sample[3]
                    do_flip_flag = sample[4]

                    # Half of the samples are have a boolean for flipping the image
                    if do_flip_flag:
                        image = cv2.flip(image, 1)
                        angle = -1.0*angle
                    images.append(image)
                    angles.append(angle)
                    throttles.append(throttle)

                # Convert to numpy array for Keras
                X_volumes.append(np.array(images))
                y_volumes.append(np.column_stack((angles, throttles)))
            X_train = np.array(X_volumes)
            y_train = np.array(y_volumes)   
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE, time_steps=TIME_STEPS)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, time_steps=TIME_STEPS)

# Test generator
count = 0
for X,y in generator(train_samples, BATCH_SIZE):
    count = count+1
    if count >= 2:
        break
print('Generator test: X.shape = {}, y.shape = {}'.format(X.shape, y.shape))
#print(y[0:5])

# # Define the neural net
# model = Sequential()
# # Preprocess incoming data, centered around zero with small standard deviation 
# model.add(Lambda(lambda x: x/127.5 - 1.,
#         input_shape=(row, col, ch),
#         output_shape=(row, col, ch)))
# Crop to focus on the part of the image containing the road
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# NVIDIA architechture
# model.add(TimeDistributed(Convolution2D(24,5,5, subsample=(2,2), activation='relu')))
# model.add(TimeDistributed(Convolution2D(36,5,5, subsample=(2,2), activation='relu')))
# model.add(TimeDistributed(Convolution2D(48,5,5, subsample=(2,2), activation='relu')))
# model.add(TimeDistributed(Convolution2D(64,3,3, activation='relu')))
# model.add(TimeDistributed(Convolution2D(64,3,3, activation='relu')))
# model.add(Flatten())

# # model = Sequential()
# # model.add(TimeDistributed(cnn))
# model.add(LSTM(256, return_sequences=True))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(2))

# model.compile(loss='mse', optimizer='adam')

# model = buildNVIDIAModell()

# volumes_per_batch = BATCH_SIZE / TIME_STEPS

model = buildLSTMModel(TIME_STEPS, cameraFormat=(row, col, ch), verbosity=1)


# Train model
start_time = time.time()
model.fit_generator(train_generator, steps_per_epoch= 
            len(train_samples)/BATCH_SIZE-1, validation_data=validation_generator, 
            validation_steps=floor(len(validation_samples)/BATCH_SIZE), epochs=3)
end_time = time.time()
print('Trained model in {:.2f} seconds'.format(end_time-start_time))

# Save the model
print("Saving model weights and configuration file.")
model.save('model_LSTM.h5')
