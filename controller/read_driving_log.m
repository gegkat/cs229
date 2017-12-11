% clear all; close all; clc;
clear all;

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


return

% filter
sa_filt = zeros(size(sa));
alpha = 0.1;
for i = 2:length(sa)
%     sa_filt(i) = sa_filt(i) + alpha * (sa(i) - sa_filt(i-1));
%     sa_filt(i) = alpha * (sa(i-1) + sa(i));
    sa_filt(i) = alpha*sa(i) + (1-alpha)*sa_filt(i-1);

end
figure; hold all; 
plot(sa, '-x')
plot(sa_filt, '-o')

figure; hold all; 
plot(sort(sa))
plot(sort(sa_filt))
