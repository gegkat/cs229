% clear all; 
close all; 
clc;

% train_file = '2017_11_17_slow/driving_log.csv';
train_file = '2017_12_02_smooth_30mph_long/driving_log.csv';

test_file = 'NVIDIA_2017-12-07_17-18-12_smooth_30mph_long/model_NVIDIA.h5_20mph.telem';
test_file = 'LSTM_time_3_epochs_20_throttle_2017-12-09_19-17-17/model_LSTM_3.h5_reverse.telem';
test_file = 'LSTM_time_3_epochs_20_throttle_2017-12-09_19-17-17/model_LSTM_3.h5_12mph.telem';

% Parse log files
train = parse_log_file(train_file);
test = parse_log_file(test_file);

test.steering = -test.steering;

% Load waypoints
waypoints = load('lake_track_waypoints.csv');

% Calculate crossrange and downrange
train = crdr(train, waypoints);
test = crdr(test, waypoints);

% Pull out a single lap
start_dr = test.dr(1); 
end_dr = 1136.84;
train_lap = find_lap(train, start_dr, end_dr);
test_lap = find_lap(test, start_dr, end_dr);

% Plot steering aug input histogram
[steering_augmented] = steering_input_hist(train.steering);

% plot everything as cdf
flds = fields(test); 
for i = 1:length(flds)
    fld = flds{i};
    figure; hold all; 
    cdf_plot(test.(fld), '-', 'DisplayName', 'test')
    cdf_plot(train.(fld), '-', 'DisplayName', 'train')
    ylabel(fld); 
end

% Plot everying from 1 lap vs dr
flds = fields(test); 
for i = 1:length(flds)
    fld = flds{i};
    figure; hold all; 
    plot(test_lap.dr, test_lap.(fld), '.-', 'DisplayName', 'test')
    plot(train_lap.dr, train_lap.(fld), '.-', 'DisplayName', 'train')
    xlabel('Down Range')
    ylabel(fld); 
end

tilefigs;


