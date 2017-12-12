function [steering_augmented] = steering_input_hist(steering)

f = figure;
set(f, 'Units', 'Pixels', 'OuterPosition', [226   434  1140 492]);

subplot(121)
histogram(steering, 20)
xlabel('Normalized Steering Input')
ylabel('# of occurences')
xlim([-1 1])
title('Raw training data')

steering_augmented = [steering; steering+0.2; steering-0.2];
steering_augmented = [steering_augmented; -steering_augmented];
subplot(122)
histogram(steering_augmented, 20)
xlabel('Normalized Steering Input')
ylabel('# of occurences')
xlim([-1 1])
title('Augmented training data')
savepdf('steer_train')


end

