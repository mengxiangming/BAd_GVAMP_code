% Code for paper:"Bilinear Adaptive Generalized Adaptive Vector Approximate
% Message Passing", IEEE Access, 2018.
% Test main code for Compressed Sensing with Matrix Uncertainty
% Code written by Xiangming Meng and Jiang Zhu
% Email: mengxm11@gmail.com, jiangzhu16@zju.edu.cn
% 2018, Sep. 27

clear;
close all; 
clc;

rng(1) % random seed 
n = 256;         % signal dimension
rate = 1;        % measurement ratio

NumBits = 1;     % quantization bits
 
prior_pi = 0.1;  % sparse ratio of signal 
prior_mean = 0;  % mean of nonzero singal 
prior_var = 1;   % var of nonzero singal
Afro2 = n;
SNR = 40;
global  dampFac T tol 
dampFac = 1;
tol = 1e-10;
T = 100;  % maximum number of iterations

m = ceil(rate*n);
tau = zeros(m,1);
Q = 10;
T_LMMSE = 1;
T_VN_denoising = 2;

MC = 5; % Monte Carlo simulation times

dMSEb_all = zeros(MC,T);
dMSEc_all = zeros(MC,T);
dMSEb_oral= zeros(MC,T);

dMSE_c_oracle_all= zeros(MC,T);
dMSE_b_oracle_all= zeros(MC,T);

% averaged over  MC realizatons
for mc = 1:MC
    K = 10;
    supp = randperm(n,K);
    x = zeros(n,1);
    x(supp) = prior_mean + sqrt(prior_var)*randn(K,1);
    A0 = sqrt(20)*randn(m,n);
    b = randn(Q,1);
 
    AQ = zeros(m,n);
    Ai = zeros(m,n,Q);
    A_b = zeros(m,Q);
    for i = 1:Q
        Ai(:,:,i) = randn(m,n);
        AQ = AQ+b(i)*Ai(:,:,i);
        A_b(:,i) = Ai(:,:,i)*x;
    end
    A = A0+AQ;

    z = A*x;
    wvar = (z'*z)*10^(-SNR/10)/m;
    w = sqrt(wvar)*randn(m,1);
    
    % Quantization interval
    nLevels = 2^NumBits-1;
    delta =  (max(z)-min(z))/(2^NumBits);
  
    % Quantize measurements
    if NumBits < inf
        y = bpdq_quantize(z+w,NumBits,delta);
    else
        y = z+w; 
    end
    
    [~, ~, dMSE_oracle_c, ~] = BAd_GVAMP_A_known(A, y, T_LMMSE, T_VN_denoising, x,b, NumBits,delta);
    
    [~, dMSE_oracle_b] = BAd_GVAMP_c_known( Ai, A0, y, T_LMMSE, T_VN_denoising, x,b, NumBits,delta);

    [x_hat_1k, x_hat_var_1k, dMSE_c, dMSE_b] = BAd_GVAMP( Ai, A0, y, T_LMMSE, T_VN_denoising, x,b, NumBits,delta);

    dMSE_c_oracle_all(mc,:) = dMSE_oracle_c;
    dMSE_b_oracle_all(mc,:) = dMSE_oracle_b;
%      
    mmse_c = dMSE_c(end)
    oracle_mmse_c = dMSE_oracle_c(end) 
    mmse_b = dMSE_b(end)
    oracle_mmse_b = dMSE_oracle_b(end)

    dMSEc_all(mc,:) = dMSE_c;
    dMSEb_all(mc,:) = dMSE_b; 
    
end

dMSE_b_oracle_all(isnan(dMSE_b_oracle_all)) = 0;
dMSE_c_oracle_all(isnan(dMSE_c_oracle_all)) = 0;
dMSEc_all(isnan(dMSEc_all)) = 0;
dMSEb_all(isnan(dMSEb_all)) = 0;

figure(1)
subplot(1,2,1)
plot(1:T,median(dMSEc_all,1),'-b*',1:T,median(dMSE_c_oracle_all,1),'--ro');
legend('dMMSE of c,BAd-GVAMP','dMMSE of c,oracle')
title(strcat('n = ',num2str(n),',ratio = ',num2str(rate),',Quantize = ',num2str(NumBits),' bit(s)'))

subplot(1,2,2)
plot(1:T,median(dMSEb_all,1),'-b*',1:T,median(dMSE_b_oracle_all,1),'--ro');
legend('MMSE of b,BAd-GVAMP','dMMSE of b,oracle')

title(strcat('n = ',num2str(n),',ratio = ',num2str(rate),',Quantize = ',num2str(NumBits),' bit(s)'))

















