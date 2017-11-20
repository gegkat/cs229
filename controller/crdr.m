function [cr, dr] = crdr(v, wp1, wp2)

v1 = wp2 - wp1; 

v2 = v -  wp1; 

theta = angle_between_vectors(v1', v2');

cr = dimwise_norm(v2')*sin(theta);

dr = dimwise_norm(v2')*cos(theta);

end