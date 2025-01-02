%% Ridwan Hussain - Adaptive Algorithm PSet 4
clc; clear all; close all;

%% Preliminary Analysis
fprintf("Preliminary Analysis\n");
A_1 = [-1.5 -2.5; 
        0.5  0.5];
A_2 = [-3 -5; 
        1  1];

C = [1 0];

Q_x = [0.1 0.0; 
       0.0 0.1];
Q_y = 0.05;

eig_A1 = eig(A_1); eig_A2 = eig(A_2);
fprintf("A_1 is stable while A_2 is unstable (eigenvalues inside vs. outside the unit circle).\n");

% Observability Matrix
N = size(A_1, 1); p = size(C, 1);
O_1(1, :) = C; O_2(1, :) = C;
for n = 2:N
    O_1(n, :) = C * A_1^(n-1);
    O_2(n, :) = C * A_2^(n-1);
end

fprintf("The rank of the observability matrix for A_1 is %d (the expected rank is %d).\n", rank(O_1), N);
fprintf("The rank of the observability matrix for A_2 is %d (the expected rank is %d).\n", rank(O_2), N);

% N states (2), m inputs (2), p outputs (1)
% [X,   K, L] = idare(A,    B,  Q,   R,   S,           E     );
  [K_1, ~, ~] = idare(A_1', C', Q_x, Q_y, zeros(N, p), eye(N));
  [K_2, ~, ~] = idare(A_2', C', Q_x, Q_y, zeros(N, p), eye(N));

%% Kalman Filter
fprintf("\nKalman Filter\n");
% Algorithm Variables
N_iter = 100;
x = zeros(N, N_iter);         % x_{n}       = x(:, N+1)    , starting from n = 0, drop first point
y = zeros(1, N_iter);         % y_{n}       = y(N)         , starting from n = 0, drop last point
x_pred = zeros(N, N_iter);    % xhat(n|n-1) = x_pred(:, N) , starting from n = 1, drop last point
x_est = zeros(N, N_iter);     % xhat(n|n)   = x_est(:, N+1), starting from n = 1, drop first point
K_pred = zeros(N, N, N_iter); % Khat(n|n-1) = K_pred(:, N) , starting from n = 1, drop last point
K_est = zeros(N, N, N_iter);  % Khat(n|n)   = K_est(:, N+1), starting from n = 1, drop first point

% Initial Values
x(:, 1) = [1; 0];                      
x_pred(:, 1) = [1; 0];              
K_pred(:, :, 1) = 0.1*eye(N);     

for n = 1:N_iter+1
   if n < 51
        [x(:, n+1), y(n)] = gen_xy(x(:, n), A_1, C, Q_x, Q_y);
   else 
        [x(:, n+1), y(n)] = gen_xy(x(:, n), A_2, C, Q_x, Q_y);
   end
end
for n = 1:(N_iter/2)                    % Using A_1 for first half
    [K_est(:, :, n+1), K_pred(:, :, n+1), x_est(:, n+1), x_pred(:, n+1)] ...
       = Kalman(K_pred(:, :, n), x_pred(:, n), A_1, C, Q_x, Q_y, y(n+1));
end
for n = (N_iter/2)+1:N_iter             % Using A_2 for second half
    [K_est(:, :, n+1), K_pred(:, :, n+1), x_est(:, n+1), x_pred(:, n+1)] ...
       = Kalman(K_pred(:, :, n), x_pred(:, n), A_2, C, Q_x, Q_y, y(n+1));
end

x = x(:, 2:N_iter+1);               % Removing initial point
x_est = x_est(:, 2:N_iter+1);       % Removing initial point
K_est = K_est(:, :, 2:N_iter+1);    % Removing initial point

x_pred = x_pred(:, 1:N_iter);       % Removing final point
K_pred = K_pred(:, :, 1:N_iter);    % Removing final point

K_1_diff = sqrt(sum(sum((K_pred - K_1).^2, 1), 2)); 
K_1_diff = reshape(K_1_diff, [size(K_1_diff, 3) 1]);
K_2_diff = sqrt(sum(sum((K_pred - K_2).^2, 1), 2)); 
K_2_diff = reshape(K_2_diff, [size(K_2_diff, 3) 1]);

figure();
plot(1:N_iter, K_1_diff);
title("K(n+1,n) vs. K_1 Ideal");
xlabel("N_{iter}"); ylabel("Difference");
xlim([1 N_iter]); xline(51);

figure();
plot(1:N_iter, K_2_diff);
title("K(n+1,n) vs. K_2 Ideal");
xlabel("N_{iter}"); ylabel("Difference");
xlim([1 N_iter]); xline(51);

fprintf("Part a: When viewing the different graphs, seeing how the switch occurs when N_iter = 51, it looks like the algortihm switches immediately.\n");
fprintf("It only needs two total iterations to reach a stable state, so the Kalman filter handles the switch and unstable data very well.\n");

% Part b
figure();
hold on;
plot(x(1, 1:N_iter), x(2, 1:N_iter));            % True Trajectory
plot(x_est(1, 2:end), x_est(2, 2:end));          % Estimatee
plot(x_pred(1, 1:N_iter), x_pred(2, 1:N_iter));  % Prediction
title("True, Estimated, and Predicted Trajectory (Row 1 vs. Row 2)");
legend("x(1, n) vs. x(2, n)", "x_{est}(1, n) vs. x_{est}(2, n)", "x_{pred}(1, n) vs. x_{pred}(2, n)");
xlabel("Row 1"); ylabel("Row 2");

% Part c
figure();
hold on;
plot(x(1, (N_iter/2+1):(N_iter)), x(2, (N_iter/2+1):(N_iter)));
title("x(1,n) vs. x(2,n) from N_{iter}/2+1 to N_iter");
legend("x(1,n) vs. x(2, n)");
xlabel("Row 1"); ylabel("Row 2"); 

x_pred_diff = sqrt(sum((x-x_pred).^2, 1));
x_est_diff  = sqrt(sum((x-x_est).^2, 1));

figure();
hold on;
plot(1:N_iter, x_pred_diff);
plot(1:N_iter, x_est_diff);
title("Difference for x_{pred} & x_{est} b/w x vs. N_{iter}");
xlabel("N_{iter}"); ylabel("Difference");
legend("x_{pred}", "x_{est}");
xlim([1 N_iter]);

% Part d
e_pred = x - x_pred;
e_est = x - x_est;

Khat_pred_pre =  (e_pred(:, 1:N_iter/2)   * e_pred(:, 1:N_iter/2)')/N_iter/2;
Khat_pred_post = (e_pred(:, (N_iter/2+1):end) * e_pred(:, (N_iter/2+1):end)')/N_iter/2;

Khat_est_pre =  (e_est(:, 1:N_iter/2)   * e_est(:, 1:N_iter/2)')/N_iter;
Khat_est_post = (e_est(:, (N_iter/2+1):end) * e_est(:, (N_iter/2+1):end)')/N_iter;

Khat_pred_pre_diff_1  = sqrt(sum(sum((K_1 - Khat_pred_pre).^2, 1), 2))
Khat_pred_pre_diff_2  = sqrt(sum(sum((K_2 - Khat_pred_pre).^2, 1), 2));

Khat_pred_post_diff_1 = sqrt(sum(sum((K_1 - Khat_pred_post).^2, 1), 2));
Khat_pred_post_diff_2 = sqrt(sum(sum((K_2 - Khat_pred_post).^2, 1), 2))

DeltaK_pre = Khat_pred_pre - Khat_est_pre;
DeltaK_post = Khat_pred_post - Khat_est_post;

DeltaK_pre_eig = eig(DeltaK_pre)
DeltaK_post_eig = eig(DeltaK_post)

%% Function
function [x_nplus1, y_n] = gen_xy(x_n, A, C, Q_x, Q_y)
    x_nplus1 = A * x_n + sqrt(Q_x) * randn(2,1);
    y_n      = C * x_n + sqrt(Q_y) * randn(1);
end

function [K_est, K_next, x_est, x_next] = Kalman(K_prev, x_prev, A, C, Q_x, Q_y, y)
    R      = ( (C * K_prev * C') + Q_y)^-1;
    G      = (K_prev * C' * R);
    alpha  = y - (C * x_prev);
    x_est  = x_prev + (G * alpha);
    x_next = A * x_est;
    
    term   = eye(size(C, 2)) - G * C;
    K_est  = term * K_prev * term' + G * Q_y * G'; 
    K_next = (A * K_est * A') + Q_x;
end