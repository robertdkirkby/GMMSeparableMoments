% GMM estimation of a stochastic volatility model of interest rates

% Import interest rate data, daily data from 1955 to 2007
t1 = datetime(1955,1,1);
t2 = datetime(2007,12,31); % 2007 as want to avoid negative interest rates as these cause a problem for this model
USinterestrates_3month=getFredData('DTB3',t1,t2);
% If you don't have getFredData() just search and download a copy

r_t=USinterestrates_3month.Data; % r_t
% drop nans (I ignore if this is problematic, as this is really just an
% example about GMM, not about interest rates. I assume they are days that
% the market was closed?)
notnan_rt=(~isnan(r_t));
r_t=r_t(notnan_rt);

taxis=USinterestrates_3month.Dates(~isnan(r_t));
taxis=taxis(2:end); % drop first one due to model containing a lag term.

delta_r_t=r_t(2:end)-r_t(1:end-1); % Delta r_t
r_t_lag=r_t(1:end-1); % r_{t-1}
r_t=r_t(2:end); % drop first obs as using lag in model (note: r_t is not used below)

T = length(delta_r_t);
length(taxis) % double-check, should be same length as T

%% First, just graph the data

fig1=figure(1);
subplot(1,2,1); plot(taxis,r_t_lag)
title('r_t')
subplot(1,2,2); plot(taxis,delta_r_t) 
title('\Delta r_{t+1}')


%% Estimate the model in the first stage with identity weighting matrix

estparams0   = [0.1; 0.1; 0.1; 1.0]; % initial guess for parameter vector (alpha,beta,sigma,gamma)

W = eye(4,4); % first-iteration, use identity matrix as weighting matrix

momentcondn_stochvol=@(estparams) momentcondnfn_stochvol(estparams,delta_r_t,r_t_lag,1);

GMMobjfn=@(estparams) momentcondn_stochvol(estparams)'*W*momentcondn_stochvol(estparams);

[estparams_iter1,fval]=fminsearch(GMMobjfn,estparams0);

%% For two-iteration GMM, we now need to estimate Omega, so we can use W=Omega^(-1)
% Because time-series are autocorrelated, we need a HAC (heteroskedasticity
% and autocorrelation robust) estimator. We use Newey-West estimator.

L = 5; % max lags for the Newey-West estimate 
g=momentcondnfn_stochvol(estparams_iter1,delta_r_t,r_t_lag,0); % [4xT matrix, the sample average moment function evaluated at each observation]

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

GMMobjfn2=@(estparams) momentcondn_stochvol(estparams)'*Weff*momentcondn_stochvol(estparams);

[estparams_iter2,fval]=fminsearch(GMMobjfn2,estparams_iter1); % note, use estparams_iter1 as initial guess (should be faster)

fprintf('Two-iteration efficient GMM gives estimates: \n')
fprintf('alpha=%1.4f \n', estparams_iter2(1))
fprintf('beta =%1.4f \n', estparams_iter2(2))
fprintf('sigma=%1.4f \n', estparams_iter2(3))
fprintf('gamma=%1.4f \n', estparams_iter2(4))


