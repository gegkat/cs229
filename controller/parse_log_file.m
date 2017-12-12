function [data] = parse_log_file(fname, type)


if ~exist('type', 'var')
    [~, ~, ext] = fileparts(fname);
    switch ext
        case '.csv'
            type = 'training';
        case '.telem'
            type = 'testing';
        otherwise
            error('Please specify whether %s is a training or testing file', fname)
    end
end

switch type
    case 'training'
        cols = {'img_center', 'img_right', 'img_left', ...
            'steering', 'throttle', 'brake', ...
            'speed', 'x', 'z', 'heading'};
        data = read_training(fname, cols);
    case 'testing'
        cols = {'steering', 'throttle', 'speed', 'x', 'z', 'heading'};
        data = read_testing(fname, cols);
    case 'model_test'
        cols = {'img', 'steer_label', 'steer_pred', ...
            'throttle_label', 'brake_label', 'throttle_pred'};
        data = read_training(fname, cols);
    otherwise
        error('Did not recognize type %s', type)
end



end

function data = read_training(fname, cols)
        
f = fileread(fname); 
lines = regexp(f, '\n', 'split');

% remove blank lines at end of file
while isempty(lines{end})
    lines(end) = [];
end

raw_data = zeros(length(lines), length(cols)); 
for i = 1:length(lines)
    curr = regexp(lines{i}, ',', 'split');
    raw_data(i,:) = str2double(curr); 
end

for i = 1:length(cols)
    data.(cols{i}) = raw_data(:,i); 
end

end

function data = read_testing(fname, cols)
        
raw_data = load(fname); 

for i = 1:length(cols)
    data.(cols{i}) = raw_data(:,i); 
end

data.steering = data.steering / 25; 
data.heading = data.heading*180/pi;
data.heading = 360 - data.heading + 90;
i = data.heading > 360; 
data.heading(i) = data.heading(i)  - 360; 

end