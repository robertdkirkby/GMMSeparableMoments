% GMM estimation of a Normal distribution
% Normal dist: N(mu,sigma^2)
% Use GMM to estimate mu and sigma by targeting the mean and standard deviation

% GMM theory tells us any weighting matrix is consistent.
% Using inverse of covariance matrix (of moments) as the weighting matrix is optimal.

% N=number of observations
% M=2 is number of moment
% k=2 is number of parameters

%% First, declare the true parameter values, then create our 'observed data'
mu=2; % 'true' value for mu
sigma=0.7; % 'true' value for sigma

N=10^3; % Number of observations
obsdata=2+0.7*randn(N,1);

trueparams=[mu; sigma];

%% Now, first thing we do to calculate the moments of our data
datamoment_mean=mean(obsdata); 
datamoment_var=var(obsdata);

datamomentvec=[datamoment_mean; datamoment_var];

%% Let's start with GMM using identity matrix as the weighting matrix
estparams0=[1;0.2]; % initial guess

W=eye(2,2); % Use identity as weighting matrix

% Note that for a normal distribution, we can trivially get the model
% moments as a function of the parameters to be estimated.
ModelMoments=@(estparams) [estparams(1); estparams(2)^2]; % the mean and variance

GMMobjfn=@(estparams) (datamomentvec-ModelMoments(estparams))'*W*(datamomentvec-ModelMoments(estparams));

[estparams,fval]=fminsearch(GMMobjfn,estparams0);

%% Efficient GMM
% Efficient GMM using the inverse of the covariance matrix of the data moments.
xbar=mean(obsdata);
Datamomentfn=@(x) [x; (x-xbar).^2];
E_Datamomentfn=mean(Datamomentfn(obsdata'),2);
Omega=zeros(2,2);
for ii=1:N
    Omega=Omega+[Datamomentfn(obsdata(ii))-E_Datamomentfn]*[Datamomentfn(obsdata(ii))-E_Datamomentfn]'; % Moment condition at each observation
end
Omega=Omega/N;

% Store for later comparison
Omega1=Omega;

Weff=Omega^(-1);

GMMobjfn_Eff=@(estparams) (datamomentvec-ModelMoments(estparams))'*Weff*(datamomentvec-ModelMoments(estparams));

[estparams_effGMM,fval_effGMM]=fminsearch(GMMobjfn_Eff,estparams0);


%% Two-iteration GMM (redundant, just to show that it does give exact same as doing efficient GMM directly)
% I do mean and variance as the two targets

% First iteration, just use W=I, which will give a consistent estimate of the parameters
W_I=eye(2,2); % Use identity as weighting matrix
GMMobjfn_W_I=@(estparams) (datamomentvec-ModelMoments(estparams))'*W_I*(datamomentvec-ModelMoments(estparams));

[estparams_iter1,fval_iter1]=fminsearch(GMMobjfn_W_I,estparams0);
% Second iteration, based on estparams_iter1, we can get covar matrix of
% [M_d-M_m], and then use inverse of this as weighting matrix

% Now we need to estimate the covar matrix of the moment conditions, given estparams_iter1
% Our moments were mean and variance
% So what we want to compute is
GMMmomentfn=@(x) [x-estparams_iter1(1); (x-xbar).^2-estparams_iter1(2)^2];
E_GMMmomentfn=mean(GMMmomentfn(obsdata'),2);
Omega=zeros(2,2);
for ii=1:N
    Omega=Omega+[GMMmomentfn(obsdata(ii))-E_GMMmomentfn]*[GMMmomentfn(obsdata(ii))-E_GMMmomentfn]'; % Moment condition at each observation
end
Omega=Omega/N;

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

% Note: because we have two moments for two parameters the weighting matrix
% is actually largely unimportant here anyway. So we are more just
% illustrating how it is (sometimes) possible to implement efficient GMM directly, rather than
% via two-iteration efficient GMM

