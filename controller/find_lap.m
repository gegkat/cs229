function [data] = find_lap(data, start_dr, end_dr)

dr = data.dr; 
dr = dr - start_dr;
neg = dr < 0; 
dr(neg) = dr(neg) + end_dr;
data.dr = dr; 

i1 = find(dr < 1, 1, 'first'); 
i2 = find(diff(dr(i1:end)) < -600, 1, 'first') - 1;
if isempty(i2); i2 = length(dr) - 1; end
idx = i1:(i1 + i2); 

flds = fields(data);
for i = 1:length(flds)
    data.(flds{i}) = data.(flds{i})(idx); 
end

end

function [idx] = find_lap_idx(dr, start_idx)

if ~exist('start_idx', 'var')
    [~, start_idx] = min(dr);
end

start = dr(start_idx); 
i1 = find(dr < start, 1, 'first'); 

if isempty(i1)
    idx = start_idx:length(dr); 
    return
end

i2 = find(dr(i1:end) > start, 1, 'first'); 

if isempty(i2)
    idx = start_idx:length(dr); 
    return
end

idx = start_idx:(i1 + i2); 
end

