%% Ridwan Hussain - Adaptive Algorithm PSet 2
clc; clear all; close all;

%% SVD and MUSIC / MVDR Spectra
fprintf("SVD and MUSIC/MVDR Spectra\n");
d_lambda = 0.5; 
alpha_dB = [0, -4, -8]; noise_dB = -12; 
% 1st row are thetas, 2nd row are phis
AOAs = [15*pi/180, 20*pi/180, 30*pi/180; 30*pi/180, 40*pi/180, -40*pi/180];

min_val = -10; max_val = 10; X = 1; Y = 1; Z = 0;
% Column 1 is x, Column 2 is y, Column 3 is z
sensor_locations = gen_locations(min_val, max_val, X, Y, Z);
% N snapshots, M sensors, L sensors
N = 100; M = size(sensor_locations, 1); L = size(AOAs, 2);

[S, A, V, X, alphas, noise] = gen_SAVX_matrix(M, L, N, d_lambda, AOAs, alpha_dB, noise_dB, sensor_locations);

% Part a
svd_X(:) = svd(X);
figure();
stem(svd_X);
title("SVD Plot of X");
ylabel("SVD Values");
xlabel("Index");
dropoff_svd = svd_X(4)/svd_X(3);
fprintf("Part a: The dropoff between sigma_4 and sigma_3 is %f\n", dropoff_svd);

% Part b
[R, Rhat] = gen_corr(M, N, S, alphas, noise_dB, X);
figure();
eig_R = sort(abs(eig(R)), "descend"); % Imaginary values are neglibly small
stem(eig_R);
title("Eigenvalues Plot of X");
ylabel("Eigen Values");
xlabel("Index");
dropoff_eig = eig_R(4)/eig_R(3);
fprintf("Part b: The dropoff between eig_4 and eig_3 is %f\n", dropoff_eig);

% Part c
[Q_L, ~, ~] = svd(X); Q_L = Q_L(:, 1:L);
P_N = eye(size(Q_L, 1)) - Q_L*ctranspose(Q_L); 
source_singular_vecs(1) = norm(P_N*S(:, 1));
source_singular_vecs(2) = norm(P_N*S(:, 2));
source_singular_vecs(3) = norm(P_N*S(:, 3));
fprintf("Part c: P_N is available to print, |P_Ns(Theta_l)| = [%f, %f, %f].\n", source_singular_vecs(1), source_singular_vecs(2), source_singular_vecs(3));

% Part d
angle_step_size = 5;
thetas = (0:angle_step_size:90) * pi/180;   phis = (-180:angle_step_size:180) * pi/180; 

AOA_grid = combvec(thetas, phis); % Took naming inspiration from Azra
L = size(AOA_grid, 2);

S_grid = gen_S_grid(M, L, d_lambda, AOA_grid, sensor_locations);

MUSIC_spec = MUSIC(S_grid, P_N);

[MUSIC_spec, AOA_index] = spectrum_plot("MUSIC", MUSIC_spec, thetas, phis, AOAs);

MVDR_spec = MVDR(S_grid, Rhat);
[MVDR_spec, ~] = spectrum_plot("MVDR", MVDR_spec, thetas, phis, AOAs);

% Part e
plot_theta("MUSIC", MUSIC_spec, thetas, AOA_index(1, 2), phis);
plot_theta("MVDR", MVDR_spec, thetas, AOA_index(1, 2), phis);

% Part f
plot_phi("MUSIC", MUSIC_spec, phis, AOA_index(2, 1), thetas);
plot_phi("MVDR", MVDR_spec, phis, AOA_index(2, 1), thetas);

%% Optimal Beamforming: MVDR and GSC
% MVDR
w_MVDR = zeros(M, size(AOAs, 2));
array_response_MVDR_dB = zeros(size(thetas, 2), size(phis, 2), size(AOAs, 2));
for i = 1:size(AOAs, 2)
    w_MVDR(:, i) = (Rhat\S(:, i)) / (ctranspose(S(:, i))/Rhat*S(:, i));
    %                                 dB conversion         Array Response Formula                  Reshape to thetas and phis    
    array_response_MVDR_dB(:, :, i) = 10*log10(reshape(abs(ctranspose(w_MVDR(:, i))*S_grid).^2, [size(thetas, 2), size(phis, 2)])); 
end
plot_question3("MVDR", array_response_MVDR_dB, phis, thetas, AOAs, AOA_index);

% GSC
C_GSC = S;
g_MVDR = eye(3);
w_GSC = C_GSC*(inv(ctranspose(C_GSC)*C_GSC))*g_MVDR; % We have w_GSC = w_q
array_response_GSC_dB = zeros(size(thetas, 2), size(phis,2 ), size(AOAs, 2));
for i = 1:size(AOAs, 2)
    array_response_GSC_dB(:, :, i) = 10*log10(reshape(abs(ctranspose(w_GSC(:, i))*S_grid).^2, [size(thetas, 2), size(phis, 2)]));
end

plot_question3("GSC", array_response_GSC_dB, phis, thetas, AOAs, AOA_index);

%% Adaptive Beamforming
%% UNCOMMENT THE PLOT FUNCTIONS AT YOUR OWN RISK. THERE ARE 50+ GRAPHS
% Can also uncomment the surf plots in the plot function at your own risk...

% MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR MVDR
% LMS
C_MVDR = S(:, 1);                              % Constraint matrix for MVDR
g_MVDR = zeros(3, 1); g_MVDR(1, 1) = 1;        % We know source 1, source 2 is unconstrainted, source 3 is NULL
u_n = X;                                       % u(n) = columns of X
w_q_MVDR = C_MVDR*inv(C_MVDR'*C_MVDR)*1;       % w_q for MVDR
[C_a_MVDR, ~, ~] = svd(C_MVDR);                % C_a is calculated as the left side of the extended SVD of C

N_iter = N;                                    % N_iter is just the number of snapshots, N
mu = 0.1;                                      % Hyper parameter for step size
d_MVDR_LMS = zeros(1, N_iter); 
x_MVDR_LMS = zeros(M, N_iter);
e_MVDR_LMS = zeros(1, N_iter);
w_a_MVDR_LMS = zeros(M, N_iter);
w_n_MVDR_LMS = zeros(M, N_iter);

for n = 1:N_iter
    d_MVDR_LMS(1, n) = w_q_MVDR' * u_n(:, n);
    x_MVDR_LMS(:, n) = C_a_MVDR' * u_n(:, n);
    e_MVDR_LMS(1, n) = d_MVDR_LMS(1, n) - w_a_MVDR_LMS(:, n)'*x_MVDR_LMS(:, n);
    w_n_MVDR_LMS(:, n) = w_q_MVDR(:, 1) - C_a_MVDR*w_a_MVDR_LMS(:, n);
    w_a_MVDR_LMS(:, n+1) = w_a_MVDR_LMS(:, n) + mu*x_MVDR_LMS(:, n)*conj(e_MVDR_LMS(1, n));
end

J_MVDR_LMS = abs(e_MVDR_LMS).^2;
figure();
plot(1:N_iter, J_MVDR_LMS);
title("MVDR LMS");
xlabel("N_{iter}"); ylabel("Error");

% plotArrRes(w_q_MVDR, C_a_MVDR, w_a_MVDR_LMS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "MVDR", "LMS", "");
% The learning curve plot does converge, but it doesn't seem like the value
% it converges to is correct.

% RLS
lambda = 0.9; delta = 1;
d_MVDR_RLS   = zeros(1, N_iter);
x_MVDR_RLS   = zeros(M, N_iter);
pi_MVDR_RLS  = zeros(M, N_iter);
k_MVDR_RLS   = zeros(M, N_iter);
P_MVDR_RLS   = zeros(M, M, N_iter); P_MVDR_RLS(:, :, 1) = eye(M)/delta; % Extra point at the initial
xi_MVDR_RLS  = zeros(1, N_iter);
w_a_MVDR_RLS = zeros(M, N_iter);                                        % Extra point at the initial

for n = 1:N_iter
    d_MVDR_RLS(1, n) = w_q_MVDR' * u_n(:, n);
    x_MVDR_RLS(:, n) = C_a_MVDR' * u_n(:, n);
    pi_MVDR_RLS(:, n) = P_MVDR_RLS(:, :, n) * x_MVDR_RLS(:, n);
    k_MVDR_RLS(:, n) = 1/(lambda + x_MVDR_RLS(:, n)' * pi_MVDR_RLS(:, n)) * x_MVDR_RLS(:, n);
    P_MVDR_RLS(:, :, n+1) = 1/lambda*P_MVDR_RLS(:, :, n) - 1/lambda*k_MVDR_RLS(:, n)*x_MVDR_RLS(:, n)'*P_MVDR_RLS(:, :, n);
    xi_MVDR_RLS(1, n) = d_MVDR_RLS(1, n) - w_a_MVDR_RLS(:, n)'*x_MVDR_RLS(:, n);
    w_a_MVDR_RLS(:, n+1) = w_a_MVDR_RLS(:, n) + k_MVDR_RLS(:, n)*conj(xi_MVDR_RLS(1, n));
end

figure();
J_MVDR_RLS = abs(xi_MVDR_RLS).^2;
plot(1:N_iter, J_MVDR_RLS);
title("MVDR RLS");
xlabel("N_{iter}"); ylabel("Error");

% plotArrRes(w_q_MVDR, C_a_MVDR, w_a_MVDR_RLS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "MVDR", "RLS", "");
% The MVDR RLS AOA plots for the phis looks ok with the nulls, but the
% theta plots look very wrong. The learning curve still converges, so it
% converges to the wrong value.

% GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC GSC
% LMS
C_GSC = S;
g = zeros(3,1); g_GSC(1,1) = 1;
w_q_GSC = C_GSC*inv(C_GSC'*C_GSC)*g_GSC;
[C_a_GSC, ~, ~] = svd(C_GSC);

d_GSC_LMS = zeros(1, N_iter); 
x_GSC_LMS = zeros(M, N_iter);
e_GSC_LMS = zeros(1, N_iter);
w_a_GSC_LMS = zeros(M, N_iter);
w_n_GSC_LMS = zeros(M, N_iter);

% Source 1
source = "Source 1:";
for n = 1:N_iter
    d_GSC_LMS(1, n) = w_q_GSC(:, 1)' * u_n(:, n);
    x_GSC_LMS(:, n) = C_a_GSC * u_n(:, n);
    e_GSC_LMS(1, n) = d_GSC_LMS(1, n) - w_a_GSC_LMS(:, n)'*x_GSC_LMS(:, n);
    w_n_GSC_LMS(:, n) = w_q_GSC(:, 1) - C_a_GSC*w_a_GSC_LMS(:, n);
    w_a_GSC_LMS(:, n+1) = w_a_GSC_LMS(:, n) + mu*x_GSC_LMS(:, n)*conj(e_GSC_LMS(1, n));
end

J_GSC_LMS = abs(e_GSC_LMS).^2;
figure();
plot(1:N_iter, J_GSC_LMS);
title(source + " GSC LMS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_LMS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "LMS", source);
% The plots look great here, we have want around 0dB at each of the
% steering vector and this is what we get. The algorithm converge to the
% right value.

% Source 2
source = "Source 2:";
for n = 1:N_iter
    d_GSC_LMS(1, n) = w_q_GSC(:, 2)' * u_n(:, n);
    x_GSC_LMS(:, n) = C_a_GSC * u_n(:, n);
    e_GSC_LMS(1, n) = d_GSC_LMS(1, n) - w_a_GSC_LMS(:, n)'*x_GSC_LMS(:, n);
    w_n_GSC_LMS(:, n) = w_q_GSC(:, 1) - C_a_GSC*w_a_GSC_LMS(:, n);
    w_a_GSC_LMS(:, n+1) = w_a_GSC_LMS(:, n) + mu*x_GSC_LMS(:, n)*conj(e_GSC_LMS(1, n));
end

J_GSC_LMS = abs(e_GSC_LMS).^2;
figure();
plot(1:N_iter, J_GSC_LMS);
title(source + " GSC LMS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_LMS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "LMS", source);
% It looks like the algorithm does place nulls but not the strong nulls we
% placed before. It converges to a value, but this may not be the correct
% value.

% Source 3
source = "Source 3:";
for n = 1:N_iter
    d_GSC_LMS(1, n) = w_q_GSC(:, 2)' * u_n(:, n);
    x_GSC_LMS(:, n) = C_a_GSC * u_n(:, n);
    e_GSC_LMS(1, n) = d_GSC_LMS(1, n) - w_a_GSC_LMS(:, n)'*x_GSC_LMS(:, n);
    w_n_GSC_LMS(:, n) = w_q_GSC(:, 1) - C_a_GSC*w_a_GSC_LMS(:, n);
    w_a_GSC_LMS(:, n+1) = w_a_GSC_LMS(:, n) + mu*x_GSC_LMS(:, n)*conj(e_GSC_LMS(1, n));
end

J_GSC_LMS = abs(e_GSC_LMS).^2;
figure();
plot(1:N_iter, J_GSC_LMS);
title(source + " GSC LMS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_LMS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "LMS", source);
% Same as the case above, there are nulls as expected, but it's not the
% strong nulls we saw before. It converges to a value, but perhaps not the
% correct value.

% RLS
lambda = 0.9; delta = 1;
d_GSC_RLS   = zeros(1, N_iter);
x_GSC_RLS   = zeros(M, N_iter);
pi_GSC_RLS  = zeros(M, N_iter);
k_GSC_RLS   = zeros(M, N_iter);
P_GSC_RLS   = zeros(M, M, N_iter); P_GSC_RLS(:, :, 1) = eye(M)/delta; % Extra point at the initial
xi_GSC_RLS  = zeros(1, N_iter);
w_a_GSC_RLS = zeros(M, N_iter);                                        % Extra point at the initial

% Source 1
source = "Source 1:";
for n = 1:N_iter
    d_GSC_RLS(1, n) = w_q_GSC(:, 1)' * u_n(:, n);
    x_GSC_RLS(:, n) = C_a_GSC' * u_n(:, n);
    pi_GSC_RLS(:, n) = P_GSC_RLS(:, :, n) * x_GSC_RLS(:, n);
    k_GSC_RLS(:, n) = 1/(lambda + x_GSC_RLS(:, n)' * pi_GSC_RLS(:, n)) * x_GSC_RLS(:, n);
    P_GSC_RLS(:, :, n+1) = 1/lambda*P_GSC_RLS(:, :, n) - 1/lambda*k_GSC_RLS(:, n)*x_GSC_RLS(:, n)'*P_GSC_RLS(:, :, n);
    xi_GSC_RLS(1, n) = d_GSC_RLS(1, n) - w_a_GSC_RLS(:, n)'*x_GSC_RLS(:, n);
    w_a_GSC_RLS(:, n+1) = w_a_GSC_RLS(:, n) + k_GSC_RLS(:, n)*conj(xi_GSC_RLS(1, n));
end

figure();
J_GSC_RLS = abs(xi_GSC_RLS).^2;
plot(1:N_iter, J_GSC_RLS);
title(source + " GSC RLS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_RLS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "RLS", source);
% This works as intended, we see 0dB on the phi plots as expected and the
% theta does have a decently high null. It converges to the right value.

% Source 2
source = "Source 2:";
for n = 1:N_iter
    d_GSC_RLS(1, n) = w_q_GSC(:, 2)' * u_n(:, n);
    x_GSC_RLS(:, n) = C_a_GSC' * u_n(:, n);
    pi_GSC_RLS(:, n) = P_GSC_RLS(:, :, n) * x_GSC_RLS(:, n);
    k_GSC_RLS(:, n) = 1/(lambda + x_GSC_RLS(:, n)' * pi_GSC_RLS(:, n)) * x_GSC_RLS(:, n);
    P_GSC_RLS(:, :, n+1) = 1/lambda*P_GSC_RLS(:, :, n) - 1/lambda*k_GSC_RLS(:, n)*x_GSC_RLS(:, n)'*P_GSC_RLS(:, :, n);
    xi_GSC_RLS(1, n) = d_GSC_RLS(1, n) - w_a_GSC_RLS(:, n)'*x_GSC_RLS(:, n);
    w_a_GSC_RLS(:, n+1) = w_a_GSC_RLS(:, n) + k_GSC_RLS(:, n)*conj(xi_GSC_RLS(1, n));
end

figure();
J_GSC_RLS = abs(xi_GSC_RLS).^2;
plot(1:N_iter, J_GSC_RLS);
title(source + " GSC RLS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_RLS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "RLS", source);
% This also works GREAT! There is a strong null on the phi and a 0dB on the
% theta plots as expected. It still converges to the right value.

% Source 3
source = "Source 3:";
for n = 1:N_iter
    d_GSC_RLS(1, n) = w_q_GSC(:, 3)' * u_n(:, n);
    x_GSC_RLS(:, n) = C_a_GSC' * u_n(:, n);
    pi_GSC_RLS(:, n) = P_GSC_RLS(:, :, n) * x_GSC_RLS(:, n);
    k_GSC_RLS(:, n) = 1/(lambda + x_GSC_RLS(:, n)' * pi_GSC_RLS(:, n)) * x_GSC_RLS(:, n);
    P_GSC_RLS(:, :, n+1) = 1/lambda*P_GSC_RLS(:, :, n) - 1/lambda*k_GSC_RLS(:, n)*x_GSC_RLS(:, n)'*P_GSC_RLS(:, :, n);
    xi_GSC_RLS(1, n) = d_GSC_RLS(1, n) - w_a_GSC_RLS(:, n)'*x_GSC_RLS(:, n);
    w_a_GSC_RLS(:, n+1) = w_a_GSC_RLS(:, n) + k_GSC_RLS(:, n)*conj(xi_GSC_RLS(1, n));
end

figure();
J_GSC_RLS = abs(xi_GSC_RLS).^2;
plot(1:N_iter, J_GSC_RLS);
title(source + " GSC RLS");
xlabel("N_{iter}"); ylabel("Error");
% plotArrRes(w_q_GSC(:, 3), C_a_GSC, w_a_GSC_RLS, N_iter, S_grid, thetas, phis, AOAs, AOA_index, "GSC", "RLS", source);
% We get double nulls for the phi and theta plots as expected. It doesn't
% seem like the strong null from before, so it might not be converging to
% the correct value. 

%% Variations
% 5a: With noise_dB = 10, the SVD plot shows that the SVD values are
% non-significant. Eigven values still look good. The Music Spectrum magnitude 
% has also dropped significantly, but MVDR seems the same. The slice plots still have
% significant peaks. The MVDR, MUSIC, and GSC stuff still work (0dB and nulls)
% The learning curve looks good for MVDR/GSC LMS but not MVDR/GSC RLS (shcoking!).

% 5b: With N = 25 (~M), SVD plots and Eigen plots look good. Spectrum plots look
% ok, but the magnitude did drop by half, but there are still well defined
% peaks. There's some ripples in the theta plots. It seems some of there is
% some issues with the nulls though on the array response for MVDR. GSC
% looked great. 
% All of the algorithms still converge to no error. MVDR LMS and RLS plots
% still look wrong, and the GSC LMS plots now look iffy (not entirely wrong), 
% but GSC RLS still looks good.. 
% Also got a lot of MatLab warnings that the matrix was close to
% singularity for the MVDR array response.

% 5b: With N = 10 (<M), SVD plots and eigen value plots are still cool.
% Spectrum plot for the MUSIC spectrum looks ok (peak magnitude of about
% 200), but for MVDR it has well defined peaks but the peak value is now in
% the e-14 (compared to 0.8 before). MVDR array response looks slight off
% (one of the sources doesn't have a null), but GSC still looks great. 

%% Functions
% Sensor Array Signal Model
function locations = gen_locations(min_val, max_val, X, Y, Z)
    range = [min_val:-1 1:max_val];
    locations = zeros(1 + (X+Y+Z)*size(range, 2), 3);
    locations(1, :) = [0, 0, 0];
    index = 2;
    if (X)
        for r = 1:size(range, 2)
            locations(index, :) = [range(r), 0, 0];
            index = index + 1;
        end
    end
    if (Y)
        for r = 1:size(range, 2)
            locations(index, :) = [0, range(r), 0];
            index = index + 1;
        end
    end
    if (Z)
        for r = 1:size(range, 2)
            locations(index, :) = [0, 0, range(r)];
            index = index + 1;
        end
    end
end

function [S, A, V, X, alphas, noise] = gen_SAVX_matrix(M, L, N, d_lambda, AOAs, alphas_dB, noise_dB, sensor_locations)
    % slide 34 is critical    
    % S: M x L; columns are steering vectors    
    S = zeros([M, L]);

    for m = 1:M % generating steering vector for a given theta
        for l = 1:L %AOAs: row1 = thetas, row2 = phis, %sensor_locations 2nd index is x, y, z
            a = sin(AOAs(1, l))*cos(AOAs(2, l))*sensor_locations(m, 1) ...   % x-axis
                + sin(AOAs(1, l))*sin(AOAs(2, l))*sensor_locations(m, 2) ... % y-axis
                + cos(AOAs(1, l))*sensor_locations(m, 3);                    % z-axis

            % d is from locations, /lambda is from formula, so we have d_lambda
            S(m, l) = 1/sqrt(M) * (exp(-1j*2*pi*a*d_lambda));
        end
    end
    
    % A: L x N; coefficients for respective steering vectors, l is the source index and n snapshot
    alphas = 10.^(alphas_dB/10); % convert alpha values from dB to mag
    A = sqrt(alphas/2)' .* (randn(L, N) + 1j*randn(L, N));
    
    % V: M x N; matrix of noise samples
    noise = (1/M) .* 10^(noise_dB/10);
    V = sqrt(noise/2) .* (randn(M,N) + 1j*randn(M,N));
    
    % X: M x N
    X = S * A + V;

end

function [R, Rhat] = gen_corr(M, N, S, alphas, noise_dB, X)
    R_A = diag(alphas); 
    var = (1/M) * 10.^(noise_dB/10);
    R_V = var * eye(M, M);
    R = S*R_A*ctranspose(S) + R_V;
    
    Rhat = (1/N) * X * ctranspose(X);
end

function MUSIC_spec = MUSIC(S, P_N)
    MUSIC_spec = 1 ./ (ctranspose(S)*P_N*S);
end

function MVDR_spec = MVDR(S, Rhat)
    MVDR_spec =  1 ./ (ctranspose(S)/Rhat*S);
end

function S_grid = gen_S_grid(M, L, d_lambda, AOAs, sensor_locations)
    % slide 34 is critical    
    % S: M x L; columns are steering vectors    
    S_grid = zeros([M, L]);

    for m = 1:M % generating steering vector for a given theta
        for l = 1:L %AOAs: row1 = thetas, row2 = phis, %sensor_locations 2nd index is x, y, z
            a = sin(AOAs(1, l))*cos(AOAs(2, l))*sensor_locations(m, 1) ...   % x-axis
                + sin(AOAs(1, l))*sin(AOAs(2, l))*sensor_locations(m, 2) ... % y-axis
                + cos(AOAs(1, l))*sensor_locations(m, 3);                    % z-axis

            % d is from locations, /lambda is from formula, so we have d_lambda
            S_grid(m, l) = 1/sqrt(M) * (exp(-1j*2*pi*a*d_lambda));
        end
    end
end

function [spectrum, AOA_index] = spectrum_plot(spec_name, spectrum, thetas, phis, AOAs)
    figure();
    spectrum = diag(abs(spectrum));
    spectrum = reshape(spectrum, [size(thetas, 2) size(phis, 2)]);
    surf(phis(1, :)*180/pi, thetas(1, :)*180/pi, spectrum);
    title(spec_name + " Spectrum");
    AOA_index = zeros(size(AOAs)); % Index where phi and theta occur for a particular grid
    hold on;
    for i = 1:size(AOAs, 2)
        AOA_index(1, i) = find(thetas == AOAs(1, i), 1); AOA_index(2, i) = find(phis == AOAs(2, i), 1); 
        plot3(AOAs(2, i)*180/pi, AOAs(1, i)*180/pi, spectrum(AOA_index(1, i), AOA_index(2, i)), "-o", "MarkerFaceColor", "#B2BEB5", "MarkerEdgeColor", "#B2BEB5");
    end
    xlabel("Phis"); ylabel("Thetas"); zlabel("Magnitude");
    hold off;
end

function plot_theta(spec_name, spec, thetas, theta_val, phis)
    spec = spec(theta_val, :);
    figure();
    plot(phis*180/pi, spec);
    title(spec_name + " Spectrum \theta=" + thetas(theta_val)*180/pi + "^{o}");
    xlabel("Phis"); 
end

function plot_phi(spec_name, spec, phis, phi_val, thetas)
    spec = spec(:, phi_val);
    figure();
    plot(thetas*180/pi, spec);
    title(spec_name + " Spectrum \phi=" + phis(phi_val)*180/pi + "^{o}");
    xlabel("Thetas"); 
end

function plot_question3(name, array_response_dB, phis, thetas, AOAs, AOA_index)
    figure();
    hold on;
    for i = 1:size(AOAs, 2)
        plot(thetas*180/pi, array_response_dB(:, AOA_index(2, 1), i));
    end
    hold off;
    title(name + " AOA \phi=30^{o}");
    xlabel("Thetas"); ylabel("Magnitude (dB)");
    legend("Array Response 1", "Array Response 2", "Array Response 3");
    xline(15, "-r"); yline(0, "-b");
    
    figure();
    hold on;
    for i = 1:size(AOAs, 2)
        plot(phis*180/pi, array_response_dB(AOA_index(1, 2), :, i));
    end
    hold off;
    title(name + " AOA \theta=20^{o}");
    xlabel("Phis"); ylabel("Magnitude (dB)");
    legend("Array Response 1", "Array Response 2", "Array Response 3");
    xline(40, "-r"); yline(0, "-b");
    
    for i = 1:3
        figure();
        surf(phis*180/pi, thetas*180/pi, array_response_dB(:, :, i));
        hold on;
        title("2D Plot of " + name + ": Array Response " + i);
        xlabel("Phis"); ylabel("Thetas"); zlabel("Magnitude (dB)");
        for k = 1:3
            plot3(AOAs(2, k)*180/pi, AOAs(1, k)*180/pi, array_response_dB(AOA_index(1, k), AOA_index(2, k), i), "-o", "MarkerFaceColor", "#B2BEB5", "MarkerEdgeColor", "#B2BEB5");
        end
        hold off;
    end
end

function plot_question4(name, array_response_dB, phis, thetas, AOAs, AOA_index)
    figure();
    hold on;
    plot(thetas*180/pi, array_response_dB(:, AOA_index(2, 1)));
    hold off;
    title(name + " AOA \phi=30^{o}");
    xlabel("Thetas"); ylabel("Magnitude (dB)");
    xline(15, "-r"); yline(0, "-b");
    legend("Array Response", "\theta = 15^{\circ}", "0dB");

    figure();
    hold on;
    plot(phis*180/pi, array_response_dB(AOA_index(1, 2), :));
    hold off;
    title(name + " AOA \theta=20^{o}");
    xlabel("Phis"); ylabel("Magnitude (dB)");
    xline(40, "-r"); yline(0, "-b");
    legend("Array Response", "\phi = 40^{\circ}", "0dB");
    
    %% UNCOMMENT FOR 3D PLOTS
    % figure();
    % surf(phis*180/pi, thetas*180/pi, array_response_dB);
    % hold on;
    % title("2D Plot of " + name + ": Array Response");
    % xlabel("Phis"); ylabel("Thetas"); zlabel("Magnitude (dB)");
    % for k = 1:3
    %     plot3(AOAs(2, k)*180/pi, AOAs(1, k)*180/pi, array_response_dB(AOA_index(1, k), AOA_index(2, k)), "-o", "MarkerFaceColor", "#B2BEB5", "MarkerEdgeColor", "#B2BEB5");
    % end
    % hold off;
end

function plotArrRes(w_q, C_a, w_a, N_iter, S_grid, thetas, phis, AOAs, AOA_index, optimizer, lmsrls, source)
    w_halfN = w_q - C_a*w_a(:, round(N_iter/2 + 1));
    w_N = w_q - C_a*w_a(:, end);
    w_avg = sum(w_q - C_a*w_a(:, end-10:end), 2)/10;
    arr_res_halfN_dB = 10*log10(reshape(abs(ctranspose(w_halfN)*S_grid).^2, [size(thetas, 2), size(phis, 2)]));
    arr_res_N_dB = 10*log10(reshape(abs(ctranspose(w_N)*S_grid).^2, [size(thetas, 2), size(phis, 2)]));
    arr_res_avg_dB = 10*log10(reshape(abs(ctranspose(w_avg)*S_grid).^2, [size(thetas, 2), size(phis, 2)]));
    
    plot_question4(source + " " + optimizer + " " + lmsrls + " N_{iter} = " + N_iter, arr_res_halfN_dB, phis, thetas, AOAs, AOA_index);
    plot_question4(source + " " + optimizer + " " + lmsrls + " N_{iter} = " + round(N_iter/2 + 1), arr_res_N_dB, phis, thetas, AOAs, AOA_index);
    plot_question4(source + " " + optimizer + " " + lmsrls + " Avg", arr_res_avg_dB, phis, thetas, AOAs, AOA_index);
end