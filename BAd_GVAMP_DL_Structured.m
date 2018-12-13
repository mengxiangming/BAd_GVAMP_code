function [x_hat_1k, theta_b, dMSE_A, dMSE_x,dMSE_b] = Grvamp_EM_Ad_final_multibit_DL_structured(Ai, A0, y, A, wvar,T_LMMSE, T_VN_denoising, x_true,b_true, NumBits,delta)
% GrVAMP algorithm for one bit compressed sensing under additive Gaussian
global  T 

[m, n] = size(A0);

L = size(x_true,2);
% Perform estimation
computeMseb = @(noise) 20*log10(norm(noise(:))/norm(b_true));

%singal 
Signal_error_function =...
    @(q) 20*log10(norm(x_true -...
    q*find_permutation(x_true,q),'fro')/norm(x_true,'fro'));
dictionary_error_function =...
    @(q) 20*log10(norm(A -...
    q*find_permutation(A,q),'fro')/norm(A,'fro'));

dMSE_A = zeros();
dMSE_x = zeros();
dMSE_b = zeros();
noise_var = zeros();
% initialization
x_hat_1k= zeros(size(x_true)); 
Q = length(b_true);
theta_b = randn(Q,1);

lar_num = 1e12;
sma_num = 1e-8;

% initialization of EM parameters
AQ_est = gen_matrix(Ai,b_true,Q);
A = A0+AQ_est;
wvar_hat = 1e3;

% wvar_hat = norm(y)^2/(100+1)/length(y);

mu0 = 0;
pi_t = 0.1;
vx = 1e1;

% Initialization for nonlinear case
z_A_ext = zeros(m,L);
v_A_ext = lar_num;
gamma2k = sma_num*ones(1,L);
r2k = zeros(size(x_true));

damp = 0.8; % damping factor

xhat2k = zeros(size(x_true));
eta2k = zeros(1,L);
eta1k = zeros(1,L);
x_hat_1k = zeros(size(x_true));
x_hat_var_1k = zeros(size(x_true));
lambda = zeros(size(x_true));

m0_est = zeros(1,L);
vx_est = zeros(1,L);

y_tilde = zeros(m,L);
v_A_post = lar_num*ones(1,L);
z_A_post = zeros(m,L);
sigma2_tilde_est = ones(1,L);
for t = 1:T

    if NumBits < inf % nonlinear observations
        % obtain the equivalent linear observations
        for l = 1:L          
            [z_B_post, v_B_post] = outputUpdate(y(:,l), z_A_ext(:,l), v_A_ext*ones(m,1), sqrt(wvar_hat), NumBits,delta);
            v_B_post = mean(v_B_post);        
            sigma2_tilde_est(l) = v_B_post.*v_A_ext./(v_A_ext-v_B_post); %  
            sigma2_tilde_est(l) = lar_num*(sigma2_tilde_est(l)<0)+sigma2_tilde_est(l).*(sigma2_tilde_est(l)>0);
            sigma2_tilde_est(l) = min(sigma2_tilde_est(l),lar_num);
            sigma2_tilde_est(l) = max(sigma2_tilde_est(l),sma_num); 
            y_tilde(:,l) = sigma2_tilde_est(l).*(z_B_post./v_B_post-z_A_ext(:,l)./v_A_ext);  % 

        end   
        sigma2_tilde = mean(sigma2_tilde_est);
%         if(t>1)
%             y_tilde = (1-damp)*y_tilde_old+damp*y_tilde;
%             sigma2_tilde = (1-damp)*sigma2_tilde_old+damp*sigma2_tilde;
%         end
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
        AQ_est = gen_matrix(Ai,theta_b,Q);
        A_eq = A0+AQ_est;
        Cov_t = 0;
        for l = 1:L
            gammaw = 1/sigma2_tilde;    
            Cov = gammaw*(A_eq'*A_eq)+gamma2k(l)*eye(n);
            Cov_inv = inv(Cov);
            xhat2k(:,l) = Cov\(gammaw*A_eq'*y_tilde(:,l)+gamma2k(l)*r2k(:,l));
            eta2k(l) = n/trace(Cov_inv);
            Cov_t = Cov_t + Cov_inv;
        end

        H = zeros(Q,Q);
        beta = zeros(Q,1);
        for qi = 1:Q 
            for qj = 1:Q
                H(qi,qj) = trace(Ai(:,:,qj)'*Ai(:,:,qi)*(Cov_t+xhat2k*xhat2k'));
            end
            beta(qi) = trace(y_tilde'*Ai(:,:,qi)*xhat2k)-trace(A0'*Ai(:,:,qi)*(Cov_t+xhat2k*xhat2k'));
        end
        theta_b = H\beta;
        
        % EM learning of noise variance
        sigma2_tilde = 1/m/L*(norm(y_tilde-A_eq*xhat2k,'fro')^2+trace(A_eq*Cov_t*A_eq'));
    end

    if(t>1)
        gamma1k_new = (eta2k-gamma2k);
        r1k_new = (repmat(eta2k,n,1).*xhat2k-repmat(gamma2k,n,1).*r2k)./repmat(gamma1k_new,n,1);
        gamma1k = damp*(gamma1k_new)+(1-damp)*gamma1k_old;
        gamma1k = max(gamma1k,sma_num);
        gamma1k = min(gamma1k,lar_num);
        r1k = damp*(r1k_new) + (1-damp)*r1k_old;
    else
        gamma1k = eta2k-gamma2k;
        gamma1k = max(gamma1k,sma_num);
        gamma1k = min(gamma1k,lar_num);
        r1k = (repmat(eta2k,n,1).*xhat2k-repmat(gamma2k,n,1).*r2k)./repmat(gamma1k,n,1);
    end
    
    gamma1k_old = gamma1k;
    r1k_old = r1k;
    
    
    % denoising in the variable node
    for k1 = 1:T_VN_denoising
        % denoising step
        vr = 1./gamma1k;
        for l = 1:L
            M = 0.5*log(vr(l)./(vr(l)+vx))+0.5*r1k(:,l).^2./vr(l)-0.5*(r1k(:,l)-mu0).^2./(vr(l)+vx);
            lambda(:,l) = pi_t./(pi_t+(1-pi_t).*exp(-M));
            m_t = (r1k(:,l).*vx+vr(l).*mu0)./(vr(l)+vx);
            V_t = vr(l).*vx./(vr(l)+vx);

            x_hat_1k(:,l) = lambda(:,l).*m_t; 
            x_hat_var_1k(:,l) = lambda(:,l).*(m_t.^2+V_t)-(lambda(:,l).*m_t).^2;

            % EM learning step for the prior parameters
            eta1k(l) = 1./mean(x_hat_var_1k(:,l));
            gamma1k(l) = 1./(1./eta1k(l)+mean((x_hat_1k(:,l)-r1k(:,l)).^2));
            m0_est(l) = lambda(:,l)'*m_t./sum(lambda(:,l));
            vx_est(l) = lambda(:,l)'*((mu0-m_t).^2+V_t)./sum(lambda(:,l));
            
        end
        pi_t = mean(lambda(:));
        mu0 = mean(m0_est);
        vx = mean(vx_est);      
    end
                    
    if(t>1)
        gamma2k_new = eta1k-gamma1k;
        r2k_new = (repmat(eta1k,n,1).*x_hat_1k-repmat(gamma1k,n,1).*r1k)./repmat(gamma2k_new,n,1);

        gamma2k = damp*(gamma2k_new)+(1-damp)*gamma2k_old;
        gamma2k = max(gamma2k,sma_num);
        gamma2k = min(gamma2k,lar_num);
        r2k = damp*(r2k_new) + (1-damp)*r2k_old;
    else
        gamma2k = eta1k-gamma1k;
        gamma2k = max(gamma2k,sma_num);
        gamma2k = min(gamma2k,lar_num);
        r2k = (repmat(eta1k,n,1).*x_hat_1k-repmat(gamma1k,n,1).*r1k)./repmat(gamma2k,n,1);
    end
    
    gamma2k_old = gamma2k;
    r2k_old = r2k;
        
    if NumBits < inf % nonlinear observations
        %--- LMMSE step for calculate the extrinsic mean and variance
        AQ_est = gen_matrix(Ai,theta_b,Q);
        A_eq = A0+AQ_est;
        for l = 1:L
            gammaw = 1/sigma2_tilde;    
            Cov = gammaw*(A_eq'*A_eq)+gamma2k(l)*eye(n);
            xhat2k(:,l) = Cov\(gammaw*A_eq'*y_tilde(:,l)+gamma2k(l)*r2k(:,l));
            z_A_post(:,l) = A_eq*xhat2k(:,l);
            v_A_post(l) = 1/m*trace(A_eq/(gammaw*(A_eq'*A_eq)+gamma2k(l)*eye(n))*A_eq');%         
        end
        v_A_post_mean = mean(v_A_post);
        v_A_ext = v_A_post_mean.*sigma2_tilde./(sigma2_tilde-v_A_post_mean);
        v_A_ext = lar_num*(v_A_ext<0)+v_A_ext*(v_A_ext>0);
        v_A_ext = min(v_A_ext,lar_num);
        v_A_ext = max(v_A_ext,sma_num);
        z_A_ext = v_A_ext.*(z_A_post./v_A_post_mean - y_tilde./sigma2_tilde); 
            
        if t>1
            z_A_ext = (1-damp)*z_A_ext_old+damp*z_A_ext;
            v_A_ext = (1-damp)*v_A_ext_old+damp*v_A_ext;
        end

        z_A_ext_old = z_A_ext;
        v_A_ext_old = v_A_ext;
        
        wvar_hat = sigma2_tilde;
%         wvar_hat = wvar;
    end
 
    % compute the debiased MMSE       
        cb = theta_b'*b_true/(theta_b'*theta_b+eps);
        dMSE_b(t) = computeMseb(cb*theta_b-b_true); 
        
        dMSE_x(t) = Signal_error_function(x_hat_1k);
        
        theta_b_debiased = cb*theta_b;
        AQ_est = gen_matrix(Ai,theta_b_debiased,Q);
        A_eq = A0 + AQ_est;
        
        lamda_est = norm(A,'fro')/norm(A_eq,'fro');
        dMSE_A(t) = 20*log10(norm(lamda_est*A_eq - A,'fro')/norm(A,'fro'));

%         dMSE_A(t) = dictionary_error_function(A_eq);  

end
xx = 1;
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
 




