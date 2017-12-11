% fname = 'NVIDIA_2017-12-07_17-18-12_smooth_30mph_long/model_NVIDIA.h5.test';
fname = 'LSTM_time_3_epochs_20_throttle_2017-12-09_19-17-17/model_LSTM_3.h5.test';

center = parse_log_file([fname '.center'], 'model_test');
% right = parse_log_file([fname '.right'], 'model_test');
% left = parse_log_file([fname '.left'], 'model_test');

% figure; 
% ax(1) = subplot(311); hold all; 
% plot(center.steer_label, '-', 'DisplayName', 'label');
% plot(center.steer_pred, '-', 'DisplayName', 'prediction');
% title('center')
% ylabel('steering')
% 
% ax(2) = subplot(312); hold all; 
% plot(right.steer_label, '-', 'DisplayName', 'label');
% plot(right.steer_pred, '-', 'DisplayName', 'prediction');
% title('right')
% ylabel('steering')
% 
% ax(3) = subplot(313); hold all; 
% plot(left.steer_label, '-', 'DisplayName', 'label');
% plot(left.steer_pred, '-', 'DisplayName', 'prediction');
% title('left')
% ylabel('steering')
% 
% linkaxes(ax, 'xy')

figure; hold all; 
plot(center.steer_label, '-', 'DisplayName', 'label');
plot(center.steer_pred, '-', 'DisplayName', 'prediction');
legend toggle
ylabel('steering')
xlim([2550.20665691918          3270.35841236587])

err_center = center.steer_pred - center.steer_label;
mse_center = mean(err_center.^2)

err_left = left.steer_pred - left.steer_label;
mse_left = mean(err_left.^2)

err_right = right.steer_pred - right.steer_label;
mse_right = mean(err_right.^2)

