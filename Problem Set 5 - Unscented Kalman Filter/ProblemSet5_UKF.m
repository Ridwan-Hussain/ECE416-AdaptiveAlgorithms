%% Ridwan Hussain - Adaptive Algorithm PSet 5
clc; clear all; close all;

%% Setup
Nx = 5; 
beta_0 = 0.597983; % got it from uhlproc function

%% Implementing the UKF
fs = 10; dt = 1/fs; 

% Simulation is over 50 seconds, or 500 time steps.
N_iter = 500;

% Generating a 0 mean vector, covariance matrix I, and 2N_x + 1 sigma points
eta_n = zeros(Nx, 2*Nx+1);      % eta_{n-1} = eta_n(n)
w_n = zeros(1, 2*Nx+1);         % w_{n-1} = w_n(n)
w_n(1) = 1/3;

for n = 1:Nx
    one_n = zeros(Nx, 1); one_n(n) = 1;
    
    eta_n(:, n+1) = sqrt(Nx/(1-w_n(1)))*one_n; 
    w_n(n+1) = (1 - w_n(1)) / (2*Nx); 

    eta_n(:, n+Nx+1) = -sqrt(Nx/(1-w_n(1)))*one_n; 
    w_n(n+Nx+1) = (1 - w_n(1)) / (2*Nx);
end

x_0(:, 1) = [6400.4; 349.14; -1.8093; -6.7967; 0.6932];   % x(n-1) = x(:, n)

Q_x = diag([1e-8, 1e-8, 2.404e-5, 2.404e-5, 1e-8]); 
Q_y = diag([1, 17e-3]);

x_pred = zeros(Nx, N_iter+1); K_pred = zeros(Nx, Nx, N_iter+1);
x_pred(:, 1) = [6400; 350; -2; -7; 0.65];               % xhat(n|n-1) = x_pred(n)

K_pred(:, :, 1) = diag([1e-4, 1e-4, 1e-4, 1e-4, 1]);    % K(n, n-1) = K_pred(:, :, n)

%% Question 1
fprintf("Question 1\n");
emp_mean = sum(eta_n .* w_n, 2);             
fprintf("Emprical mean = %f\n", emp_mean);
emp_cov = (eta_n .* w_n) * eta_n';                                
emp_skew = sum((eta_n.^3 .* w_n), 2);
emp_kurt = sum((eta_n.^4 .* w_n), 2);
fprintf("Compare kurtosis to Gaussian case!\n");
kurt = Nx/(1-w_n(1));
% fprintf("Emprical mean is 0 because everything eta_n(1) is 0, eta_n(n+1)=-eta_n(n+Nx+1) with equal weights. Total sum is equal to 0.\n");
fprintf("To get Kurtosis equal to 3, we have -(Nx/3 - 1)=%f, which makes w_0 negative.\n", -(Nx/3 - 1));

%% Question 2
fprintf("\nQuestion 2\n");
[w_n2, eta_n2] = sigma_points(emp_mean, chol(emp_cov, "lower"), w_n(1));
fprintf("Can manually print out w_n vs. w_n2 and eta_n vs. eta_n2 to verify they're the same.\n");

%% Question 3
fprintf("\nQuestion 3\n");
emp_mean2 = sigma_mean(w_n2, eta_n2);
fprintf("Can manually print out emp_mean vs. emp_mean2 to verify they're the same.\n");

%% Question 4
fprintf("\nQuestion 4\n");
emp_cov = sigma_cov(w_n2, eta_n2);
fprintf("Can manually print out emp_cov vs. emp_cov2 to verify they're the same.\n");

%% Question 5
fprintf("\nQuestion 5\n");
y_meas = zeros(2, N_iter); x_meas = zeros(Nx, N_iter); x_meas(:, 1) = x_0;
weights = zeros(size(w_n, 2), N_iter); weights(:, 1) = w_n2;
x_est = zeros(Nx, N_iter);    
K_est = zeros(Nx, Nx, N_iter);

linespec1 = "--r"; linespec2 = "-b";
name1 = "Measured Pos"; name2 = "Estimated Pos";
graph1 = [linespec1 name1]; graph2 = [linespec2, name2];
xlabel1 = "Position x"; ylabel1 = "Position y";

name3 = "Measured Vel"; name4 = "Estimated Vel";
graph3 = [linespec1 name3]; graph4 = [linespec2, name4];
xlabel2 = "Velocity x"; ylabel2 = "Velocity y";

name5 = "\beta_{meas}"; name6 = "\beta_{est}";
graph5 = [linespec1 name5]; graph6 = [linespec2 name6]; 
xlabel3 = "N_{iter}"; ylabel3 = "\beta";

for loop = 1:5
    for n = 1:N_iter
        [x_pred(:, n+1), K_pred(:, :, n+1), x_est(:, n), K_est(:, :, n), w_n2] = ukf_filter(x_pred(:, n), K_pred(:, :, n), Q_x, Q_y, y_meas(:, n), n, dt, w_n2);
        [x_meas(:, n+1), y_meas(:, n+1)] = generate_ymeas(x_meas(:, n), n, dt, Q_x, Q_y);
    end

    title1 = "Loop " + loop + ": Measured vs. Estimated Position";
    labels1 = [xlabel1, ylabel1, title1];
    title2 = "Loop " + loop + ": Measured vs. Estimated Velocity";
    labels2 = [xlabel2, ylabel2, title2];
    title3 = "Loop " + loop + ": Measured vs. Estimated Beta";
    labels3 = [title3, xlabel3, ylabel3];
    
    plotGraphs([x_meas(1, :); x_meas(2, :)], graph1, [x_est(1, :); x_est(2, :)], graph2, labels1);
    plotGraphs([x_meas(3, :); x_meas(4, :)], graph3, [x_est(3, :); x_est(4, :)], graph4, labels2);
    beta_meas = beta_0 * exp(x_meas(5, 1:end-1)); beta_est = beta_0 * exp(x_est(5, :));
    plotGraph(beta_meas, graph5, beta_est, graph6, labels3);
end

%% Functions
function [x_meas, y_meas] = generate_ymeas(x_meas_prev, n, dt, Q_x, Q_y)
    x_meas = uhlprocsim(x_meas_prev, n*dt, dt, chol(Q_x, "lower"), "m");
    y_meas = uhlmeas(x_meas, n*dt, chol(Q_y, "lower"));
end

function [w, sigma_pts] = sigma_points(mu, C_chol, w_0)
    Nx = size(C_chol, 1);

    w(1) = w_0; w_tilde = (1 - w_0) / (2*Nx);
    w(2:(2*Nx+1)) = w_tilde;

    eta_0(:, 1) = zeros(Nx, 1);                              % First 0's column
    eta_0(:, 2:2+Nx-1) = 1 / sqrt(2*w_tilde)*eye(Nx);        % Positive Square
    eta_0(:, 2+Nx:2+Nx*2-1) = -1 / sqrt(2*w_tilde)*eye(Nx);  % Negative Square
    
    sigma_pts = mu + C_chol * eta_0;
end

function sigmu = sigma_mean(w, sig)         % Mean of sigma points
    sigmu = sum(sig .* w, 2);        
end

function sigcov = sigma_cov(w, sigx, sigy)  % Covariance of sigma points
    if nargin == 2                          % Auto-Covariance of sigma points
        sigy = sigx';
    elseif nargin == 3                      % Covariance of sigx and sigy
        sigy = sigy';
    else
        error("Function takes in two or three inputs, sigma_cov(w, sigx[, sigy]).\n");
    end
    sigcov = ((sigx - sigma_mean(w, sigx)) .* w) * (sigy' - sigma_mean(w, sigy'))';
end

function [xp_new, Kp_new, x_est, K_est, w_2] = ukf_filter(xp_prev, Kp_prev, Q_x, Q_y, y_meas, n, dt, w_0)
    % Variables for function use
    t = n*dt; Nx = size(xp_prev, 1);
    
    % 1: Generate new x sigma points
    [w_1, eta_xp_prev] = sigma_points(xp_prev, chol(Kp_prev, "lower"), w_0(1));

    % 2: Generate new y sigma points
    eta_yp_prev = zeros(2, Nx);
    for i = 1:size(eta_xp_prev, 2)
        eta_yp_prev(:, i) = uhlmeas(eta_xp_prev(:, i), t, zeros(2));
    end

    % 3: Calculate y predicted in the previous state
    yp_prev = sigma_mean(w_1, eta_yp_prev);

    % 4: Calculate cov matrix Kxy, Kyy
    K_xy = sigma_cov(w_1, eta_xp_prev, eta_yp_prev);
    K_yy = sigma_cov(w_1, eta_yp_prev) + Q_y;
    
    % 5: Calculate Kalman variables
    alpha = y_meas - yp_prev;
    G     = K_xy * K_yy^-1;
    x_est = xp_prev + G*alpha;

    K_est = Kp_prev - G*K_yy*G';

    % 6: Calculate estimated x sigma points
    [w_2, eta_xe] = sigma_points(x_est, chol(K_est, "lower"), w_1(1));

    % 7: Calculate predicted sigma points
    eta_xp = uhlprocsim(eta_xe, t, dt, zeros(Nx), "m"); % Use midpoint method

    % 8: Calculate final x predicted and K predicted
    xp_new = sigma_mean(w_2, eta_xp);
    Kp_new = sigma_cov(w_2, eta_xp) + Q_x;
end

function plotGraphs(data1, graph1, data2, graph2, labels)
    figure();
    hold on;
    plot(data1(1, :), data1(2, :), graph1(1), "DisplayName", graph1(2)); % --r
    plot(data2(1, :), data2(2, :), graph2(1), "DisplayName", graph2(2)); % --r
    title(labels(1)); xlabel(labels(2)); ylabel(labels(3)); 
    legend;
end

function plotGraph(beta_meas, graph1, beta_est, graph2, labels)
    figure();
    hold on;
    plot(1:size(beta_meas,2), beta_meas, graph1(1), "DisplayName", graph1(2));
    plot(1:size(beta_est,2), beta_est, graph2(1));
    title(labels(1)); xlabel(labels(2)); ylabel(labels(3)); 
    legend;
end