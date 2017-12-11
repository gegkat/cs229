function cdf_plot(val, varargin)

val = sort(val); 
x = linspace(0, 1, length(val));
plot(x, val, varargin{:}); 
xlabel('percentile')

end

