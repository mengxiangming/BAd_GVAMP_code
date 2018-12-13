% Code for paper:"Bilinear Adaptive Generalized Adaptive Vector Approximate
% Message Passing", IEEE Access, 2018. 
% Test code for Structured dictionary learning
% Code written by Xiangming Meng and Jiang Zhu
% Email: mengxm11@gmail.com, jiangzhu16@zju.edu.cn
% 2018, Sep. 27


clear;
close all; 
clc;


n = 64;         % signal dimension
rate = 1;

NumBits = 3;  % quantization bits
 
prior_pi = 0.2;
prior_mean = 0;
prior_var = 1;
Afro2 = n;
SNR = 40;

global  dampFac T tol 
dampFac = 1;
tol = 1e-10;
T = 100;  % maximum number of iterations

m = ceil(rate*n);
% L = ceil(5*n*log(n));
L = 10;

tau = zeros(m,1);
Q = n;
T_LMMSE = 1;
T_VN_denoising = 2;

MC = 1; % Monte Carlo simulation times
dMSEb_all = zeros(MC,T);
dMSEA_all = zeros(MC,T);
dMSEx_all = zeros(MC,T);

dMSEb_oral= zeros(MC,T);

dMSE_c_oracle_all= zeros(MC,T);
dMSE_b_oracle_all= zeros(MC,T);

% averaged over MC realizatons
for mc = 1:MC
    K = ceil(prior_pi*n);
    x = zeros(n,L);

    for l = 1:L
        supp = randperm(n,K);
        x(supp,l) = prior_mean + sqrt(prior_var)*randn(K,1);
    end
    
    A0 = sqrt(0)*randn(m,n);
    b = randn(Q,1);
 
    AQ = zeros(m,n);
    Ai = zeros(m,n,Q);
    
    for i = 1:Q
        Ai(:,:,i) = randn(m,n);
        AQ = AQ+b(i)*Ai(:,:,i);
    end
    A = A0+AQ;

    z = A*x;
    wvar = (norm(z,'fro')^2)*10^(-SNR/10)/m/L;
    w = sqrt(wvar)*randn(m,L);
    
    % Quantization interval
    nLevels = 2^NumBits-1;
    delta =  (max(z(:))-min(z(:)))/(2^NumBits);
  
    % Quantize measurements
    if NumBits < inf
        y = bpdq_quantize(z+w,NumBits,delta);
    else
        y = z+w; 
    end
    [x_hat, b_hat, dMSE_A, dMSE_x,dMSE_theta] = BAd_GVAMP_DL_Structured( Ai, A0, y, A,wvar, T_LMMSE, T_VN_denoising, x,b, NumBits,delta);


    dMSEA_all(mc,:) = dMSE_A;
    dMSEb_all(mc,:) = dMSE_theta; 
    dMSEx_all(mc,:) = dMSE_x; 

end

dMSEA_all(isnan(dMSEA_all)) = 0;
dMSEb_all(isnan(dMSEb_all)) = 0;

figure(1)
subplot(1,2,1)
plot(1:T,median(dMSEA_all,1),'-b*');
legend('MMSE of A,BAd_GVAMP')
title(strcat('n = ',num2str(n),',ratio = ',num2str(rate),',Quantize = ',num2str(NumBits),' bit(s)'))

subplot(1,2,2)
plot(1:T,median(dMSEx_all,1),'-b*');
legend('MMSE of X,BAd_GVAMP')

title(strcat('n = ',num2str(n),',ratio = ',num2str(rate),',Quantize = ',num2str(NumBits),' bit(s)'))


















