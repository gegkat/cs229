clear all; close all; clc;

f = fileread('driving_log.csv'); 
lines = regexp(f, '\n', 'split');

M = zeros(length(lines), 10); 
for i = 1:length(lines)
    curr = regexp(lines{i}, ',', 'split');
    M(i,:) = str2double(curr); 
end

M(:,1:3) = []; 

cols = {'steeringAngle', 'throttle', 'brake', 'Speed div 100', 'posx', 'posy', 'heading'};
for i = 1:length(cols)
    figure; 
    y = M(:,i); 
    if i == 4; y = y/100; end
    plot(y, '.-')
    title(cols{i})
end

figure; hold all; 
for i = 1:4
    y = M(:,i); 
    if i == 4; y = y/100; end
    plot(y, '.-', 'DisplayName', cols{i})
end
legend toggle

figure; hold all; 
plot(M(:,5), M(:,6), '-')
waypoints = load('lake_track_waypoints.csv');
plot(waypoints(:,1), waypoints(:,2), '-x')