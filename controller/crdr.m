function data = crdr(data, wp)

wp_dist = cumsum([0; dimwise_norm(diff(wp), 2)]);


pos = [data.x data.z];

d = zeros(size(wp, 1), size(pos, 1));
for i = 1:size(wp, 1)
    delta = pos - wp(i,:);
    d(i, :) = sum(delta.^2, 2);
end

[~, nearest] = min(d, [], 1);

% loop waypoints around
wp = [wp(end,:); wp; wp(1,:)];
wp_dist = [wp_dist(end,:); wp_dist; wp_dist(1,:)];
nearest = nearest + 1;

cr = zeros(size(nearest));
dr = zeros(size(nearest));
for i = 1:length(nearest)
    j = nearest(i); 
    [cr(i), dr(i)] = crdr_func1(pos(i,:), wp(j-1:j+1,:), wp_dist(j-1:j+1));
end

data.cr = cr;
data.dr = dr;
end

function [cr, dr] = crdr_func1(v, wp, wp_dist)

% figure; plot(wp(:,1), wp(:,2), '-x', v(1), v(2), 'o', wp(1,1), wp(1,2), 's')

[cr, dr] = crdr_func2(v, wp(2,:), wp(3,:), wp_dist(2));
if dr < 0
    [cr, dr] = crdr_func2(v, wp(1,:), wp(2,:), wp_dist(1));
end

end

function [cr, dr] = crdr_func2(v, wp1, wp2, wp_dist)

v1 = wp2 - wp1;

v2 = v -  wp1;

theta = angle_between_vectors(v1', v2');

cr = dimwise_norm(v2')*sin(theta);

dr = dimwise_norm(v2')*cos(theta) + wp_dist;

end