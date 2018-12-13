function [x_hat_1k, x_hat_var_1k, dMSE, dMSE_b] = BAd_GVAMP_A_known(A, y, T_LMMSE, T_VN_denoising, x_true,b_true, NumBits,delta)
% Code for paper:"Bilinear Adaptive Generalized Adaptive Vector Approximate
% Message Passing", IEEE Access, 2018. 
% main function for BAd-GVAMP algorithm (L = 1 case) with A known
% Code written by Xiangming Meng and Jiang Zhu
% Email: mengxm11@gmail.com, jiangzhu16@zju.edu.cn
% 2018, Sep. 27

global  T 

[m, n] = size(A);

% Perform estimation
computeMse = @(noise) 20*log10(norm(noise(:))/norm(x_true));
computeMseb = @(noise) 20*log10(norm(noise(:))/norm(b_true));


dMSE = zeros();
dMSE_b = zeros();


% initialization
x_hat_1k= zeros(n,1); 
Q = length(b_true);
theta_b = zeros(Q,1);

% initialization of EM parameters
wvar_hat = 1e1;
% wvar_hat = norm(y)^2/(100+1)/length(y);
mu0 = 0;
pi_t = 0.1;
vx = (norm(y)^2 - length(y)*wvar_hat)/norm(A, 'fro' )^2/pi_t;

% Initialization for nonlinear case
lar_num = 1e6;
sma_num = 1e-6;
z_A_ext = zeros(m,1);
v_A_ext = lar_num;
gamma2k = sma_num;
r2k = zeros(size(x_true));

damp = 0.8; % damping factor
A_eq = A;

for t = 1:T
    if NumBits < inf % nonlinear observations
        % obtain the equivalent linear observations
        [z_B_post, v_B_post] = outputUpdate(y, z_A_ext, v_A_ext*ones(m,1), sqrt(wvar_hat), NumBits,delta);
        v_B_post = mean(v_B_post);        
        sigma2_tilde = v_B_post.*v_A_ext./(v_A_ext-v_B_post); %  
        sigma2_tilde = lar_num*(sigma2_tilde<0)+sigma2_tilde.*(sigma2_tilde>0);
        sigma2_tilde = min(sigma2_tilde,lar_num);
        sigma2_tilde = max(sigma2_tilde,sma_num);
        y_tilde = sigma2_tilde.*(z_B_post./v_B_post-z_A_ext./v_A_ext);  %  

        sigma2_tilde = mean(sigma2_tilde);
%         if(t>1)
%             y_tilde = (1-damp)*y_tilde_old+damp*y_tilde;
%             sigma2_tilde = (1-damp)*sigma2_tilde_old+damp*sigma2_tilde;
%         end
% 
%         y_tilde_old = y_tilde;
%         sigma2_tilde_old = sigma2_tilde;
    else
        y_tilde = y;
        if t==1 
            sigma2_tilde = wvar_hat; % for unquantized measurements, the noise variance is initialized
        end        
    end
 
    
     % LMMSE estimation
    for k0 = 1:T_LMMSE
        gammaw = 1/sigma2_tilde;    
        Cov = gammaw*(A_eq'*A_eq)+gamma2k*eye(n);
        Cov_inv = inv(Cov);
        xhat2k = Cov\(gammaw*A_eq'*y_tilde+gamma2k*r2k);
        eta2k = n/trace(inv(Cov));
        
%         if NumBits == inf
            % EM learning of noise variance
            sigma2_tilde = 1/m*((y_tilde-A_eq*xhat2k)'*(y_tilde-A_eq*xhat2k)+trace(A_eq*Cov_inv*A_eq'));
%         end
    end

    if(t>1)
        gamma1k_new = (eta2k-gamma2k);
        r1k_new  = (eta2k*xhat2k-gamma2k*r2k)/gamma1k_new; 
        
        gamma1k = damp*(gamma1k_new)+(1-damp)*gamma1k_old;
        gamma1k = max(gamma1k,sma_num);
        gamma1k = min(gamma1k,lar_num);
        r1k = damp*(r1k_new) + (1-damp)*r1k_old;
    else
        gamma1k = eta2k-gamma2k;
        gamma1k = max(gamma1k,sma_num);
        gamma1k = min(gamma1k,lar_num);
        r1k = (eta2k*xhat2k-gamma2k*r2k)/gamma1k;
    end
    
    gamma1k_old = gamma1k;
    r1k_old = r1k;
    

    % denoising in the variable node
    for k1 = 1:T_VN_denoising
        % denoising step
        vr = 1./gamma1k;
        M = 0.5*log(vr./(vr+vx))+0.5*r1k.^2./vr-0.5*(r1k-mu0).^2./(vr+vx);
        lambda = pi_t./(pi_t+(1-pi_t).*exp(-M));
        m_t = (r1k.*vx+vr.*mu0)./(vr+vx);
        V_t = vr.*vx./(vr+vx);

        x_hat_1k = lambda.*m_t; 
        x_hat_var_1k = lambda.*(m_t.^2+V_t)-(lambda.*m_t).^2;

        % EM learning step for the prior parameters
        eta1k = 1./mean(x_hat_var_1k);
        gamma1k = 1/(1/eta1k+mean((x_hat_1k-r1k).^2));
        pi_t = mean(lambda);
        mu0 = lambda'*m_t/sum(lambda);
        vx = lambda'*((mu0-m_t).^2+V_t)/sum(lambda);
    end
                    
    if(t>1)
        gamma2k_new = eta1k-gamma1k;
        r2k_new = (eta1k.*x_hat_1k-gamma1k.*r1k)./gamma2k_new;
        
        gamma2k = damp*(gamma2k_new)+(1-damp)*gamma2k_old;
        gamma2k = max(gamma2k,sma_num);
        gamma2k = min(gamma2k,lar_num);
        r2k = damp*(r2k_new) + (1-damp)*r2k_old;
    else
        gamma2k = eta1k-gamma1k;
        gamma2k = max(gamma2k,sma_num);
        gamma2k = min(gamma2k,lar_num);
        r2k = (eta1k.*x_hat_1k-gamma1k.*r1k)./gamma2k;
    end
    
    gamma2k_old = gamma2k;
    r2k_old = r2k;    
     
    if NumBits < inf % nonlinear observations 
        %--- LMMSE step for calculate the extrinsic mean and variance
        gammaw = 1/sigma2_tilde;
        Cov = gammaw*(A_eq'*A_eq)+gamma2k*eye(n);
        xhat2k = Cov\(gammaw*A_eq'*y_tilde+gamma2k*r2k);
      
        z_A_post = A_eq*xhat2k;
        v_A_post = 1/m*trace(A_eq/(gammaw*(A_eq'*A_eq)+gamma2k*eye(n))*A_eq');%  

        v_A_ext = v_A_post.*sigma2_tilde./(sigma2_tilde-v_A_post);
        v_A_ext = lar_num*(v_A_ext<0)+v_A_ext*(v_A_ext>0);
        v_A_ext = min(v_A_ext,lar_num);
        v_A_ext = max(v_A_ext,sma_num);
        z_A_ext = v_A_ext.*(z_A_post./v_A_post-y_tilde./sigma2_tilde);

        if t>1
            z_A_ext = (1-damp)*z_A_ext_old+damp*z_A_ext;
            v_A_ext = (1-damp)*v_A_ext_old+damp*v_A_ext;
        end

        z_A_ext_old = z_A_ext;
        v_A_ext_old = v_A_ext;
        
        wvar_hat = sigma2_tilde;
    end
    
    
    %damp = max(0.1,damp*0.95);
 
    % compute the debiased MMSE
    if(NumBits==1)
        c0 = x_hat_1k'*x_true/(x_hat_1k'*x_hat_1k+eps);
        dMSE(t) = computeMse(c0*x_hat_1k-x_true);
        cb = theta_b'*b_true/(theta_b'*theta_b+eps);
        dMSE_b(t) = computeMseb(cb*theta_b-b_true);
    else 
        dMSE(t) = computeMse(x_hat_1k-x_true);
        dMSE_b(t) = computeMseb(theta_b-b_true);
    end
    
end

end



function [z_post, vz_post] = outputUpdate(y, z, mz, sigma, NumBits,delta)
% Performs output node update.
%
% NOTE: This function can potentially run into numerical erros. This is due
% to the sub-function evaluateTotalMoment, which performs integration 
% of a gaussian in some integral given by quantizer boundaries. In case
% when this inteval is far from the mean of the normal and the normal has a
% small variance moments might result in 0, although in reality they
% represent some small values, ratio of which is definetely non-zero.

% length of the signal to estimate
m = size(y, 1);

% Total effective noise (AWGN + estiamtion)
mtv = mz + (sigma^2);

% Initialize outputs

% comupte the lower and up bounds
r_low = y - delta/2;
r_low(r_low < -(2^NumBits-1/2)*delta) = -1e50;

r_up = y + delta/2;
r_up(r_up > (2^NumBits-1/2)*delta) = 1e50;
 
% complex-valued case
% ita1 = (sign(y).*z - min(abs(r_low),abs(r_up)))./sqrt(2*mtv);
% ita2 = (sign(y).*z - max(abs(r_low),abs(r_up)))./sqrt(2*mtv);
% 
% z_post = z + sign(y).*mz./sqrt(mtv).*((normpdf(ita1) - normpdf(ita2))./(normcdf(ita1) - normcdf(ita2)));
% vz_post = mz/2 - mz.^2./(2*mtv).*((ita1.*normpdf(ita1) - ita2.*normpdf(ita2))./(normcdf(ita1) - normcdf(ita2)) + ((normpdf(ita1) - normpdf(ita2))./(normcdf(ita1) - normcdf(ita2))).^2);

% real-valued case

ita1 = (sign(y).*z - min(abs(r_low),abs(r_up)))./sqrt(mtv);
ita2 = (sign(y).*z - max(abs(r_low),abs(r_up)))./sqrt(mtv);


A = normpdf(ita1) - normpdf(ita2);
B = normcdf(ita1) - normcdf(ita2);
C = ita1.*normpdf(ita1) - ita2.*normpdf(ita2);


D = A./B;

E = C./B + (A./B).^2;


Small_toc = 1e-50;
D(abs(B)<Small_toc) = - ita1(abs(B)<Small_toc);
E(abs(B)<Small_toc) = 1;

z_post = z + sign(y).*mz./sqrt(mtv).*D;
vz_post = mz - mz.^2./(mtv).*(E);
end
 




