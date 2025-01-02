%% Ridwan Hussain - Adaptive Algorithm PSet 3
clc; clear all; close all;

%% Generating the Data
N_0 = 3; M_0 = 5; M_max = 11; M = M_max;
K = 1e4;
alphas = [0.1, 0.2, 0.3]; 
P_dB = [-30, -10];

lambda = 0.9; delta = 0.01; 
w_init = zeros(M_max, 1); w_init(M_0, 1) = 1;
N_iter = [20, 50];

%% RLS & Inverse QRD-RLS
for a = 1:size(alphas, 2)
    for p = 1:size(P_dB, 2)
        for n = 1:size(N_iter, 2)
            alpha = alphas(a); % Choose alpha value to run the program with:    [1, 2, 3]
            var = P_dB(p);     % Choose variance value to run the program with: [1, 2, 3]
            iter = N_iter(n);  % Choose N_iter value:                           [20, 50]
            [var_v, N_train, len, x, v, h, y, d, A] = gen_variables(alpha, var, N_0, M_0, M_max, K, iter);
            
            [w_RLS, xi_RLS, k_RLS, P_RLS] = RLS(d, A, lambda, delta, w_init, iter, M_max);
            % plotAlgos("RLS", xi_RLS, alpha, var_v, iter); % Can plot the error values to see values do indeed converge
                                                            % Got the idea from Jeffrey
            
            [w_inv_QRD_RLS, xi_inv_QRD_RLS, k_inv_QRD_RLS, gamma_inv_QRD_RLS, P_inv_QRD_RLS] = inv_QRD_RLS(d, A, lambda, delta, w_init, iter, M_max);
            % plotAlgos("Inverse QRD RLS", xi_inv_QRD_RLS, alpha, var_v, iter);
            
            %% Comparing Results
            fprintf("For alpha = %f, variance = %f & N_iter = %d, w_f = \n", alpha, var, iter);
            final_w_RLS = w_RLS(:, end)
            final_w_inv_QRD_RLS = w_inv_QRD_RLS(:, end)            
            
            P_final = P_RLS(:, :, end);
            P_ch    = P_inv_QRD_RLS(:, :, end);
            spec_norm = norm(P_final - P_ch*P_ch');
            fprintf("The spectral norm difference is %e.\n", spec_norm);
            
            SNIR_raw = -10*log10(4*abs(alpha)^2+var_v(1, 1));
            SNIR_opt = -10*log10(var_v(1, 1));
            
            fprintf("SNIR_theoretical_raw this run = %f\n", SNIR_raw);
            fprintf("SNIR_opt this run = %f\n", SNIR_opt);
            
            x_est_RLS = zeros(1, K-10); x_est_inv_QRD_RLS = zeros(1, K-10);
            for m = 1:(K-M)
                x_est_RLS(m)                 = y(m+N_train+M_0 : m+N_train+M_0+M-1)*w_RLS(:, end);
                x_est_inv_QRD_RLS(m)         = y(m+N_train+M_0 : m+N_train+M_0+M-1)*w_inv_QRD_RLS(:, end);
            end
            
            SNIR_eq_RLS_raw         = -10*log10(sum(abs(x_est_RLS         - x(N_train+M_0:end-M)).^2, "all")/K);
            SNIR_eq_inv_QRD_RLS_raw = -10*log10(sum(abs(x_est_inv_QRD_RLS - x(N_train+M_0:end-M)).^2, "all")/K);
            fprintf("The raw SNIR value for RLS was %f, and for inv QRD RLS was %f.\n", SNIR_eq_RLS_raw, SNIR_eq_inv_QRD_RLS_raw);
            fprintf("The values are very close because the final w's for both are really close.\n");
            fprintf("These calculated SNIRs are much a bit lower than the calculated theoretical ones.\n");
            fprintf("\n\n");
        end
    end
end

%% Functions
function [var_v, N_train, len, x, v, h, y, d, A] = gen_variables(alpha, P_dB, N_0, M_0, M_max, K, N_iter)
    var_v = 10.^(P_dB/10);
    
    N_train = M_max - M_0 + N_iter - 1; % Length of training sequence
    len = N_train + M_0 + K;
    
    x = randn(1, len); x(x>=0) = 1; x(x<0) = -1; % Generates +- 1 signal
    v = randn(1, len) * sqrt(var_v(1, 1)); %% CHECK THIS THING IF DIVIDE BY SIZE OR NOT %%
    h = [zeros(1, N_0-1), alpha, 1, -alpha];
    y = filter(h, 1, x) + v;
    
    d = x(1, (M_max-M_0):(M_max-M_0+N_iter-1));
    A = toeplitz(y(M_max:-1:1), y(M_max:(M_max+N_iter-1)));
end

function [w, xi, k, P_final] = RLS(d, A, lambda, delta, w_init, N_iter, M)
    % w is a M x N_iter matrix, first col is NOT w_init
    % Each w is M x 1
    % xi is a-priori estimate
    % k is a mtrix where columns are Kalman gains
    % P_final is the final inverse covariance matrix
    
    w = zeros(M, 1); w(:, 1) = w_init;
    xi = zeros(M, 1);
    P_final(:, :, 1) = delta^(-1) * eye(M); s = zeros(M, N_iter);
    k = zeros(M, N_iter);

    u = A;
    for n = 2:N_iter
        s(:, n) = P_final(:, :, n-1) * u(:, n);
        k(:, n) = (lambda + ctranspose(u(:, n))*s(:, n))^(-1) * s(:, n);
        xi(n) = d(n) - ctranspose(w(:, n-1)) * u(:, n);
        w(:, n) = w(:, n-1) + k(:, n)*ctranspose(xi(n));
        P_final(:, :, n) = lambda^(-1) * ( P_final(:, :, n-1) - k(:, n)*ctranspose(s(:, n)) );
    end

end

function [w, xi, k, gamma_inv_sqrt, P_ch] = inv_QRD_RLS(d, A, lambda, delta, w_init, N_iter, M)
    w(:, 1) = w_init;
    xi = zeros(1, N_iter);
    P_ch(:, :, 1) = chol(delta^(-1) * eye(M), "lower"); 
    u = A; 
    k = zeros(M, N_iter);
    prearray = zeros(M+1, M+1, N_iter); 
    postarray = zeros(M+1, M+1, N_iter);
    gamma_inv_sqrt = zeros(N_iter, 1);
    D = zeros(M+1, M+1, N_iter);

    for n = 2:N_iter
        prearray(:, :, n) = [1 lambda^(-0.5)*ctranspose(u(:, n))*P_ch(:, :, n-1); zeros(M, 1) lambda^(-0.5)*P_ch(:, :, n-1)];
        postarray(:, :, n) = ( qr(prearray(:, :, n)') )';
        D(:, :, n) = diag(diag(sign(postarray(:,:,n))));
        postarray(:, :, n) = postarray(:, :, n) * D(:, :, n);
    
        P_ch(:, :, n) = postarray(2:end, 2:end, n);
        gamma_inv_sqrt(n) = postarray(1, 1, n);
        k(:, n) = gamma_inv_sqrt(n)^(-1) * (postarray(2:end, 1, n));
    
        xi(:, n) = d(:, n) - ctranspose(w(:, n-1))*u(:, n);
        w(:, n) = w(:, n-1) + k(:, n)*conj(xi(:, n));
    end

end

function plotAlgos(name, xi, alpha, var_v, N_iter)
    figure();
    xi = reshape(xi, N_iter, 1);
    plot(abs(xi));
    title(name + " Plot where \alpha=" + alpha + ", \sigma^2_v=" + var_v + ", N_{iter}=" + N_iter);
    xlabel("N_{iter}"); ylabel("\xi");
end