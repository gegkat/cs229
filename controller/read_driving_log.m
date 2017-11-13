clear all; close all; clc;

f = fileread('driving_log.csv'); 
lines = regexp(f, '\n', 'split');

M = zeros(length(lines), 7); 
for i = 1:length(lines)
    curr = regexp(lines{i}, ',', 'split');
    M(i,:) = str2double(curr); 
end

M(:,1:3) = []; 
M(:,4) = M(:,4)/100; % Convert speed to approx 1 scale 

cols = {'steeringAngle', 'throttle', 'brake', 'Speed div 100'};
for i = 1:length(cols)
    figure; 
    plot(M(:,i), '.-')
    title(cols{i})
end

figure; hold all; 
for i = 1:length(cols)
    plot(M(:,i), '.-', 'DisplayName', cols{i})
end
legend toggle