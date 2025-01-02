%% Ridwan Hussain - Adaptive Algorithm PSet 1
clc; clear all; close all;

%% Initilization
tic
poles = [[0.9, 0.8]; [0.95, 0.8]; [0.95, -0.9]];
for p = 1:size(poles, 1)
    invFilter(p) = zpk([], poles(p, :), 1);
    [~, a(p, :)] = zp2tf([], poles(p, :), 1);
end
M = [2; 4; 10]; var_v = 1; 

%% Data Analysis
fprintf("FOR DATA ANALYSIS:\n");
N_iter = 1000; 
% Poles are rows, M values are columns for the cell array
for p = 1:size(poles, 1)
    for m = 1:size(M, 1)        
        [K1(p, m), K2(p, m), r_m(p, m), w_opt(p, m), p_max(p, m), N_init(p, m), N_0(p, m), v(p, m), x(p, m), X(p, m), R_L(p, m), PSD(p), ...
            eigen_val(p), min_evals(p), max_evals(p), x_L(p, m), v_L(p, m), X_L(p, m), M_L(p, m), R_hat(p, m), r_0_hat(p, m)] ...
        = DataAnalysis(p, M(m), poles(p, :), N_iter, a(p, :));
    end
    fprintf("\n");
end

%% LMS 
fprintf("\nFOR LMS:");
convergence_const = 1; % Constant I made up for different mu step sizes where 
                       % convergence_const = 1 is the stability condition.
N_iter = 10000;        % midasj > 1, avg curve above Jmin, M too big, overfit
for p = 1:size(poles, 1)
    for m = 1:size(M, 1)
        if p == 1 && m == 1            % Pole Pair 1
            convergence_const = 1.5; 
        elseif p == 1 && m == 2
            convergence_const = 0.55;
        elseif p == 1 && m == 3
            convergence_const = 0.35;
        elseif p == 2 && m == 1        % Pole Pair 2
            convergence_const = 3;
        elseif p == 2 && m == 2
            convergence_const = 1.6; 
        elseif p == 2 && m == 3
            convergence_const = 0.65; 
        elseif p == 3 && m == 1        % Pole Pair 3
            convergence_const = 0.03;
        elseif p == 3 && m == 2
            convergence_const = 0.025;
        elseif p == 3 && m == 3
            convergence_const = 0.02;
        end
        [w_LMS(p, m), w_end_LMS(p, m), e_LMS(p, m), J_LMS(p, m), misadj_LMS(p, m)] = LMS(p, M(m), N_iter, ...
            cell2mat(N_init(p, m)), a(p, :), cell2mat(PSD(p)), convergence_const, var_v, cell2mat(w_opt(p, m)));
    end
end

fprintf("\nQuestions from LMS Section:\n");
fprintf("How does mu affect ROC and midasj?\n"); 
fprintf("The large the mu value, the faster ROC but the bigger the misadj. Smaller mu values lead to slower ROC but smaller misdadj.\n");

fprintf("How does M affect ROC, misadj, and the mu value used?\n"); 
fprintf("M makes the entire system more unstable, which requires larger mu and more iterations for convergence. ");
fprintf("Also, when M=/=2, it makes it harder for w2 specifically to converge to the correct value.\n");
fprintf("When M>2, since we are in 2D space for this problem, we end up overfitting the data once the dimensions of the vector is greater than 2.\n");
fprintf("This is why it seems like the final output is much more unstable for the M=4 and M=10 states.\n");

fprintf("How does the poles affect the ROC, misadj, mu, and M?\n");
fprintf("polepair2 was the most unstable one (close to unit circle), since the poles were closest to the unit circle.\n");
fprintf("polepair1 was alright, it was converging, but was a bit off.\n");
fprintf("polepair 3 was the best since it had two poles super far away from each other.\n");

fprintf("Does w converge to w_opt for any run?") 
fprintf("The closest to converging was polepair 3 when M=2, with the error being in the ~10^-3 range.\n");
fprintf("The values for the difference between w_opt and w_actual (averaged across the 100 runs) are printed when the function is called.\n");
fprintf("The numerical difference between w_actual and w_sum can be seen at the end of each of the LMS runs.\n")

%% RLS
fprintf("\nFOR RLS:");
N_iter = 80; % I SHOULD LOWER THIS VALUE AS NEED BE!!!!

for p = 1:size(poles, 1)
    for m = 1:size(M, 1)                                     
        [s_RLS(p, m), k_RLS(p, m), e_RLS(p, m), w_RLS(p, m), w_actual_RLS(p, m), P_RLS(p, m), J_RLS(p, m), ...
            misadj_RLS(p, m)] = RLS(p, M(m), N_iter, cell2mat(N_init(p, m)), a(p, :), var_v, cell2mat(w_opt(p, m)));
    end
end

toc
%% Functions
function [x, v, N_0] = randgen(N_iter, M, N_init, a)
    % # of rows = M + 1, # of cols = N_iter
    N_0 = N_iter + M;
    v = randn(N_0 + N_init, 1); % Pretend data was whitened already
    v = v(N_init + 1: end); 
    x = filter(1, a, v); % Filter data to channel sounding parametes (poles are related to a)
end
function [K1, K2, r_m, w_opt, p_max, N_init, N_0, v, x, X, R_L, PSD, eigen_val, min_evals, max_evals, x_L, v_L, X_L, M_L, R_hat, r_0_hat] = DataAnalysis(polepair, M, pole, N_iter, a)
    if M == 2
        fprintf("For Pole Pair " + polepair + "\n");
    end

    K1 = pole(1) / ( (1-pole(1)^2) * (pole(1)*(1+pole(2)^2) - pole(2)*(1+pole(1)^2)) );
    K2 = pole(2) / ( (1-pole(2)^2) * (pole(2)*(1+pole(1)^2) - pole(1)*(1+pole(2)^2)) );
    w_opt = [-a(2); -a(3); zeros(M-2, 1)]; % Using a(2) and a(3) since a(1) = z^0 coefficient

    % Task 1
    p_max = max(pole);
    N_init = ceil(log(0.01) / log(p_max));    

    [x, v, N_0] = randgen(N_iter, M, N_init, a);

    % My attempt without toeplitz (it works)
    % X = zeros(M, N_iter);
    % for j = 1:M+1
    %     X_2(j, :) = (x(1+M-j+1:N_iter+M-j+1))';
    % end
    
    % First column goes from x(M+1) to x(1) decreasing
    % The rows go from M+1 to N_0
    X = toeplitz(x(M+1:-1:1), x(M+1:N_0));

    L = 20;
    r_m = (K1 * pole(1).^(0:L) + K2 * pole(2).^(0:L))';

    R_L = toeplitz(r_m);
    r_L = R_L(1, :);

    w = linspace(-pi, pi, 1000);
    PSD = 1 ./ ((1-pole(1).*exp(1j*w)).*(1-pole(1)./exp(1j*w)).*(1-pole(2).*exp(1j*w)).*(1-pole(2)./exp(1j*w)));
    PSD = real(PSD);

    eigen_val = eig(R_L);
    min_eval = min(eigen_val); max_eval = max(eigen_val);

    min_evals = zeros(L+1, 1); max_evals = zeros(L+1, 1);

    for m = 1:L+1
            eigen_vals = eig(R_L(1:m, 1:m)); 
            min_evals(m) = min(eigen_vals);
            max_evals(m) = max(eigen_vals);
    end
    PSD_min = min(PSD); PSD_max = max(PSD);

    M_L = L; 
    [x_L, v_L, N_0L] = randgen(N_iter, M_L, N_init, a);

    X_L = toeplitz(x_L(M_L+1:-1:1), x_L(M_L+1:N_0L));
    K = size(X_L, 2); % The number of columns makes this unbiased
    R_hat = 1/K*X_L*transpose(X_L);
    r_L_hat = R_hat(1, :);

    if M == 2 % PSD, eigen values, and correlation graphs don't change with different values of M
        figure();
        plot(w, PSD);
        title("PSD Plot for Pole Pair " + polepair);
        xlabel("Frequency (rad/s)"); ylabel ("Magnitude");
        xlim([-pi pi]);

        figure();
        stem(size(eigen_val,1):-1:1, eigen_val);
        title("Eigenvalue Stem Plot for Pole Pair " + polepair);
               
        fprintf("Minimum eigen value is %f\n", min_eval);
        fprintf("The minimum PSD value is %f and the minimum eigenvalues for m [1, L+1] are [", PSD_min);
        fprintf("%g, ", min_evals(1:end-1)); % Source: https://www.mathworks.com/matlabcentral/answers/423743-how-to-use-fprintf-to-display-vector
        fprintf("%g].\n", min_evals(end));
        fprintf("We can verify that PSD min is less than or equal to the minimum eigenvalues.\n");
    
        fprintf("Maximum eigen value is %f\n", max_eval);
        fprintf("The maximum PSD value is %f and the maximum eigenvalues for m [1, L+1] are [", PSD_max);
        fprintf("%g, ", max_evals(1:end-1));
        fprintf("%g], and the maximum PSD value is %f.\n", max_evals(end));
        fprintf("We can verify that PSD max is less than or equal to the maximum eigenvalues.\n");

        figure();
        hold on; 
        stem(r_L); stem(r_L_hat);
        title("Correlation Graph Comparison for Pole Pair " + polepair + " and M = " + M);
        legend("r_L", "r_{L}hat");
        hold off;
    end
    
    % Task 2, Question 5: We scale alphaX where alpha = 1/sqrt(K_0)
    K_0 = 100;
    alpha = 1/sqrt(K_0);
    Deltaw = w(2)-w(1);
    r_0_hat = 1/(2*pi) * sum(PSD*Deltaw);

    K1 = {K1}; K2 = {K2}; r_m = {r_m}; w_opt={w_opt}; p_max = {p_max}; N_init = {N_init}; N_0 = {N_0}; v = {v}; x = {x}; 
    X = {X}; R_L = {R_L}; PSD = {PSD}; eigen_val = {eigen_val}; min_evals = {min_evals}; max_evals = {max_evals}; 
    x_L = {x_L}; v_L = {v_L}; X_L = {X_L}; M_L = {M_L}; R_hat = {R_hat}; r_0_hat = {r_0_hat};
end
function [w, w_end, e, J, misadj] = LMS(polepair, M, N_iter, N_init, a, PSD, convergence_const, var_v, w_opt)
    fprintf("\nFor Pole Pair " + polepair + ", M = " + M + ", and convergence constant = " + convergence_const + ".\n");
    % Jmin = variance or innovations, v[n]

    % For all the runs, mu = 1/Smax has always been a stable parameter,
    % where there are no random spikes in the learning curve.

    % LMS PARAMETERS
    K_0 = 100; 
    w = zeros(M, 1, 3); % w = zeros(M, 1, 3, K_0);
    w_end = zeros(M, K_0);
    e = zeros(M, 1, K_0); J = zeros(N_iter+M, 1);
    mu_opt = 1/max(PSD) * convergence_const; % optimal mu values from trial and error
    mu_test1 = 1/max(PSD) * convergence_const * 0.5;
    mu_test2 = 1/max(PSD) * convergence_const * 0.05;
    mu = [mu_opt mu_test1 mu_test2];

    for k = 1:K_0
        [d, ~, ~] = randgen(N_iter, M, N_init, a);
        for n = 1:(N_iter+M)
            u = zeros(M, 1);
            for i = 1:M
                if (n-i) >= 1
                    u(i, 1) = d(n-i);
                end % Otherwise default the value to 0 from the zeros matrix
            end
            e(n, 1, k) = d(n, 1) - w(:, n, 1)'*u; 
            w(:, n+1, 1) = w(:, n, 1) + mu(1)*u*conj(e(n, 1, k));
            e(n, 2, k) = d(n, 1) - w(:, n, 2)'*u;
            w(:, n+1, 2) = w(:, n, 2) + mu(2)*u*conj(e(n, 2, k));
            e(n, 3, k) = d(n, 1) - w(:, n, 3)'*u;
            w(:, n+1, 3) = w(:, n, 3) + mu(3)*u*conj(e(n, 3, k));
        end
        w_end(:, k) = w(:, end, 1);
    end

    J(:, 1) = (1 / K_0) * sum((abs(e(:, 1, :)).^2), 3);
    J(:, 2) = (1 / K_0) * sum((abs(e(:, 2, :)).^2), 3);
    J(:, 3) = (1 / K_0) * sum((abs(e(:, 3, :)).^2), 3);

    figure();
    plot((1:N_iter+M), J(:, 1), (1:N_iter+M), J(:, 2), (1:N_iter+M), J(:, 3));
    title("Learning Curve Average Error Graph for LMS Pole Pair " + polepair + " and M = " + M);
    yline(var_v, "-", "Jmin");
    legend("\mu_{opt}", "\mu_{0.5}", "\mu_{0.05}", "Jmin");
    ylabel("Abs Err"); xlabel("Iteration");
    xlim([0 N_iter+M]); ylim([0 10]);

    misadj_range = 50;
    misadj = sum(J((end-misadj_range):end, :), "all") / var_v / misadj_range;
    fprintf("The misadjustment is equal to %f (taken from the last %d points) which is greater than 1.\n", misadj, misadj_range);

    if M == 2
        figure();
        plot((1:N_iter+M), e(:, 1, 2).^2);
        title("Learning Curve One Run Error Graph for LMS Pole Pair " + polepair + " and M = " + M);
        yline(var_v, "-", "Jmin");
        legend("mu_{opt}", "Jmin");
        ylabel("Abs Err"); xlabel("Iteration");
        xlim([0 N_iter+M]);

        figure();
        w_difference = abs(w(:, 1:end-1, 1) - w_opt).^2; % Need to remove last index since algo calculates w(n+1) value
        plot((1:N_iter+M), (w_difference(1, :, 1)), (1:N_iter+M), (w_difference(2, :, 1)));
        title("Learning Curve One Run w Difference Graph for LMS Pole Pair " + polepair + " and M = " + M);
        legend("w_1", "w_2");
        xlabel("Iteration");
        xlim([0 N_iter+M]);
    end
    wsum = sum((w_end-w_opt).^2, 2)/K_0;
    w1_diff = wsum(1);
    w2_diff = wsum(2);
    fprintf("The difference between w and w_opt is roughly %f and %f.\n", w1_diff, w2_diff);

    w = {w}; w_end = {w_end}; e = {e}; J = {J}; misadj = {misadj};
end
function [s, k, e, w, w_end, P, J, misadj] = RLS(polepair, M, N_iter, N_init, a, var_v, w_opt)
    fprintf("\nFor Pole Pair " + polepair + " and M = " + M + "\n");
    
    % RLS PARAMETERS!!!!
    K_0 = 100; w = zeros(M, 1); w_end = zeros(M, K_0);
    e = zeros(M, 1, K_0); J = zeros(N_iter+M, 1);
    lambda = 0.9; delta = 1; 
    P(:, :, 1) = delta^(-1) * eye(M); s = zeros(M, N_iter+M);

    for k_0 = 1:K_0
        [d, ~, ~] = randgen(N_iter, M, N_init, a);
        for n = 2:(N_iter+M)
            u = zeros(M, 1);
            for i = 1:M
                if (n-i) >= 1
                    u(i, 1) = d(n-i);
                end % Otherwise default the value to 0 from the zeros matrix
            end
            s(:, n) = P(:, :, n-1) * u;
            k = (lambda + ctranspose(u)*s(:, n))^(-1) * s(:, n);
            e(n, k_0) = d(n) - ctranspose(w(:, n-1)) * u;
            w(:, n) = w(:, n-1) + k*ctranspose(e(n, k_0));
            P(:, :, n) = lambda^(-1) * ( P(:, :, n-1) - k*ctranspose(s(:, n)) );
            if (find(eig(P(:, :, n)) <= 0))
                fprintf("P(n) was NOT Positive Definite in this loop! Returned RLS prematurely.");
                s = {s}; k = {k}; e = {e}; w = {w}; w_end = {w_end}; P = {P}; J = {J}; misadj = {0};
                return
            end
        end
        w_end(:, k_0) = w(:, end);
    end

    for n = 1:(N_iter+M)
        J(n) = (1 / K_0) * sum(abs(e(n, :)).^2); 
    end

    figure();
    plot((1:N_iter+M), J);
    title("Learning Curve Average Error Graph for RLS Pole Pair " + polepair + " and M = " + M);
    yline(var_v, "-", "Jmin");
    legend("RLS", "Jmin");
    ylabel("Abs Err"); xlabel("Iteration");
    xlim([0 N_iter+M]); ylim([0 10]);

    misadj_range = 20;
    misadj = sum(J((end-misadj_range):end), "all") / var_v / misadj_range;
    fprintf("The misadjustment is equal to %f (taken from the last %d points) which is greater than 1.\n", misadj, misadj_range);

    wsum = sum(w_end, 2)/(K_0);
    w1_diff = wsum(1) - w_opt(1);
    w2_diff = wsum(2) - w_opt(2);
    fprintf("The difference between w and w_opt is roughly %f and %f.\n", w1_diff, w2_diff);

    s = {s}; k = {k}; e = {e}; w = {w}; w_end = {w_end}; P = {P}; J = {J}; misadj = {misadj};
end