clear all;
close all; 
clc;

telem_output = load('drive.out'); 

flds = {'z', 'steeringAngle', 'speed', 'throttle', 'x', 'heading'};
for i = 1:length(flds)
    data.(flds{i}) = telem_output(:,i);
end

data.heading = data.heading*180/pi;

for i = 1:length(flds)
    figure; plot(data.(flds{i}))
    title(flds{i})
end


waypoints = load('lake_track_waypoints.csv');
figure; hold all;
plot(data.x, data.z)
plot(waypoints(:,1), waypoints(:,2), '-x')