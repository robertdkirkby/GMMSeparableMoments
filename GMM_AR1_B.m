% GMM estimation of AR(1) process
% y_t=rho*y_{t-1}+e, e~N(0,sigma^2)
% We estimate [c,rho,sigma]

% Difference to previous example with an AR(1): we use same model, same
% true values of parameters, but because we have a different choice of
% GMM moment conditions we now can only implement efficient GMM via
% two-iteration efficient GMM.

%% First, declare the true parameter values, then create our 'observed data'
rho=0.8; % 'true' value for mu
sigma=0.3; % 'true' value for sigma

T=10^3; % Number of observations
burnin=100; % time-series, so simulation has burnin to eliminate influence of initial value
obsdata=zeros(T+burnin,1);
for tt=2:T+burnin
    obsdata(tt)=rho*obsdata(tt-1)+sigma*randn(1,1);
end
obsdata=obsdata(burnin+1:end); % drop the burnin

trueparams=[rho; sigma];

%% Estimate the model in the first stage with identity weighting matrix
estparams0=[0.5; 0.2]; % initial guess

W = eye(3,3); % first-iteration, use identity matrix as weighting matrix

momentcondn_AR1_B=@(estparams) momentcondnfn_AR1_B(estparams,obsdata,1);

GMMobjfn=@(estparams) momentcondn_AR1_B(estparams)'*W*momentcondn_AR1_B(estparams);

[estparams_iter1,fval]=fminsearch(GMMobjfn,estparams0);


%% For two-iteration GMM, we now need to estimate Omega, so we can use W=Omega^(-1)
% Because time-series are autocorrelated, we need a HAC (heteroskedasticity
% and autocorrelation robust) estimator. We use Newey-West estimator.

L = 5; % max lags for the Newey-West estimate 
g=momentcondnfn_AR1_B(estparams_iter1,obsdata,0); % [4xT matrix, the sample average moment function evaluated at each observation]

% Variance-Autocovariance Matrix
% Start with tau=0 term
Omega = g*g'; % 4x4 matrix (summing over t) (note, divide by T is done later)
for tau=1:L
    % For each tau, calculate S_{t,tau}
    S_t_tau = g(:,(tau+1):end)*g(:,1:(end-tau))'; % This is g_t*g_{t,tau}'
    % Now add it to Omega (twice, for tau and -tau, the latter is just the transpose), and apply NW kernel
    Omega = Omega + (1-tau/(L+1))*(S_t_tau + S_t_tau');
end
Omega = Omega/T;

%% 2nd-iteration, use W=Omega^(-1) as weighting matrix

Weff=Omega^(-1);

GMMobjfn2=@(estparams) momentcondn_AR1_B(estparams)'*Weff*momentcondn_AR1_B(estparams);

[estparams_iter2,fval]=fminsearch(GMMobjfn2,estparams_iter1); % note, use estparams_iter1 as initial guess (should be faster)

fprintf('Two-iteration efficient GMM gives estimates: \n')
fprintf('rho  =%1.4f \n', estparams_iter2(1))
fprintf('sigma=%1.4f \n', estparams_iter2(2))

