% GMM estimation of AR(1) process
% y_t=rho*y_{t-1}+e, e~N(0,sigma^2)
% We estimate [c,rho,sigma]

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

%% Now, first thing we do to calculate the moments of our data
datamoment_var=var(obsdata);
datamoment_1stautocovar=cov(obsdata(2:end),obsdata(1:end-1)); 
datamoment_2ndautocovar=cov(obsdata(3:end),obsdata(1:end-2)); 

datamomentvec=[datamoment_var; datamoment_1stautocovar(1,2); datamoment_2ndautocovar(1,2)];
% target moments: variance, 1st-order autocovar, 2nd-order autocovar

%% Let's start with GMM using identity matrix as the weighting matrix
estparams0=[0.5; 0.2]; % initial guess

W=eye(length(datamomentvec),length(datamomentvec)); % Use identity as weighting matrix

% Note that for an AR(1) process, we can trivially get the model
% moments as a function of the parameters to be estimated.
% Var(y)=(sigma^2)/(1-rho^2)
% Note: n-th order autocovar=[(sigma^2)/(1-rho^2)]*rho^n
% So 1st order autocovar=rho*((sigma^2)/(1-rho^2))
% And 2nd order autocovar=(rho^2)*((sigma^2)/(1-rho^2))
ModelMoments=@(estparams) [(estparams(2)^2)/(1-estparams(1)^2); estparams(1)*((estparams(2)^2)/(1-estparams(1)^2)); (estparams(1)^2)*((estparams(2)^2)/(1-estparams(1)^2))]; % variance, 1st-order autocovariance,  2nd-order autocovariance

GMMobjfn=@(estparams) (datamomentvec-ModelMoments(estparams))'*W*(datamomentvec-ModelMoments(estparams));

[estparams,fval]=fminsearch(GMMobjfn,estparams0);


%% Efficient GMM
% Efficient GMM using the inverse of the covariance matrix of the data moments.
ybar=mean(obsdata(3:end)); % drop first two so get same number of obs as we will for the 2nd-order autocovar
% moments: variance, 1st-order autocovar, 2nd-order autocovar
Datamomentfn=@(y) [(y(3:end)-ybar).^2, (y(3:end)-ybar).*(y(2:end-1)-ybar), (y(3:end)-ybar).*(y(1:end-2)-ybar)]'; % 3-by-(T-2) [-2 is due to two lags]
E_Datamomentfn=@(y,tau) mean(Datamomentfn(y(tau+1:end)),2);

% Because time-series are autocorrelated, we need a HAC (heteroskedasticity
% and autocorrelation robust) estimator. We use Newey-West estimator.

L = 5; % max lags for the Newey-West estimate 
% Variance-Autocovariance Matrix
% Start with tau=0 term
Omega=[Datamomentfn(obsdata)-E_Datamomentfn(obsdata,0)]*[Datamomentfn(obsdata)-E_Datamomentfn(obsdata,0)]';  % 3x3 matrix (summing over t) (note, divide by T is done later)
for tau=1:L
    % For each tau, calculate S_{t,tau}
    S_t_tau=[Datamomentfn(obsdata(tau+1:end))-E_Datamomentfn(obsdata,tau)]*[Datamomentfn(obsdata(1:end-tau))-E_Datamomentfn(obsdata,tau)]'; % Moment condition at each observation        
    % Now add it to Omega (twice, for tau and -tau, the latter is just the transpose), and apply NW kernel
    Omega = Omega + (1-tau/(L+1))*(S_t_tau + S_t_tau');
end
Omega=Omega/T;

% Store for later comparison
Omega1=Omega;

Weff=Omega^(-1);

GMMobjfn_Eff=@(estparams) (datamomentvec-ModelMoments(estparams))'*Weff*(datamomentvec-ModelMoments(estparams));

[estparams_effGMM,fval_effGMM]=fminsearch(GMMobjfn_Eff,estparams0);


%% Two-iteration GMM (redundant, just to show that it does give exact same as doing efficient GMM directly)

% First iteration, just use W=I, which will give a consistent estimate of the parameters
W_I=eye(3,3); % Use identity as weighting matrix
GMMobjfn_W_I=@(estparams) (datamomentvec-ModelMoments(estparams))'*W_I*(datamomentvec-ModelMoments(estparams));

[estparams_iter1,fval_iter1]=fminsearch(GMMobjfn_W_I,estparams0);
% Second iteration, based on estparams_iter1, we can get covar matrix of
% [M_d-M_m], and then use inverse of this as weighting matrix

% Now we need to estimate the covar matrix of the moment conditions, given estparams_iter1
% Our moments were mean and variance
% So what we want to compute is
GMMmomentfn1=@(y,estparams) (y(3:end)-ybar).^2 -(estparams(2)^2)/(1-estparams(1)^2);
GMMmomentfn2=@(y,estparams) (y(3:end)-ybar).*(y(2:end-1)-ybar) - estparams(1)*((estparams(2)^2)/(1-estparams(1)^2));
GMMmomentfn3=@(y,estparams) (y(3:end)-ybar).*(y(1:end-2)-ybar) - (estparams(1)^2)*((estparams(2)^2)/(1-estparams(1)^2));
GMMmomentfn=@(y,estparams) [GMMmomentfn1(y,estparams),GMMmomentfn2(y,estparams),GMMmomentfn3(y,estparams)]'; % 3-by-(T-2) [-2 is due to two lags]
E_GMMmomentfn=@(tau,y,estparams) mean(GMMmomentfn(y(tau+1:end),estparams),2);

GMMmomentfn(obsdata,estparams_iter1)
E_GMMmomentfn(0,obsdata,estparams_iter1)

% Because time-series are autocorrelated, we need a HAC (heteroskedasticity
% and autocorrelation robust) estimator. We use Newey-West estimator.

L = 5; % max lags for the Newey-West estimate 
% Variance-Autocovariance Matrix
% Start with tau=0 term
Omega=[GMMmomentfn(obsdata,estparams_iter1)-E_GMMmomentfn(0,obsdata,estparams_iter1)]*[GMMmomentfn(obsdata,estparams_iter1)-E_GMMmomentfn(0,obsdata,estparams_iter1)]';  % 3x3 matrix (summing over t) (note, divide by T is done later)
for tau=1:L
    % For each tau, calculate S_{t,tau}
    S_t_tau=[GMMmomentfn(obsdata(tau+1:end),estparams_iter1)-E_GMMmomentfn(tau,obsdata,estparams_iter1)]*[GMMmomentfn(obsdata(1:end-tau),estparams_iter1)-E_GMMmomentfn(tau,obsdata,estparams_iter1)]'; % Moment condition at each observation        
    % Now add it to Omega (twice, for tau and -tau, the latter is just the transpose), and apply NW kernel
    Omega = Omega + (1-tau/(L+1))*(S_t_tau + S_t_tau');
end
Omega=Omega/T;

% Store for later comparison
Omega2=Omega;

Wtwoiter=Omega^(-1);

GMMobjfn_Wtwoiter=@(estparams) (datamomentvec-ModelMoments(estparams))'*Wtwoiter*(datamomentvec-ModelMoments(estparams));

[estparams_iter2,fval_iter2]=fminsearch(GMMobjfn_Wtwoiter,estparams_iter1);


%% Notice that Omega1=Omega2, Wtwoiter=Weff, and estparams_effGMM=estparams_iter2
Omega1
Omega2
Wtwoiter
Weff
estparams_effGMM
estparams_iter2

% Note: we three moments for two parameters the weighting matrix
% is important here. So we are illustrating how it is (sometimes) possible to implement 
% efficient GMM directly, rather than via two-iteration efficient GMM




