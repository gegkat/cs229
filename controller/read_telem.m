clear all;
close all; 
clc;

a = load('drive.out3'); 

flds = {'x', 'speed', 'heading', 'steering_angle', 'z',' throttle'}
flds = {'brake', 'throttle', 'heading', 'x', 'z', 'speed'};


flds = {'heading', 'throttle', 'steering_angle', 'speed', 'z', 'x'}

flds = {'steeringAngle', 'throttle', 'heading', 'z', 'x', 'speed'};
for i = 1:6
    data.(flds{i}) = a(:,i);
    figure; plot(data.(flds{i}))
    title(flds{i})
end
