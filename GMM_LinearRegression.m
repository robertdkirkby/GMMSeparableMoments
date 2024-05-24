% GMM example using linear regression

%% First, declare the true parameter values, then create our 'observed data'
alpha=4; % 'true' value for intercept
beta=2; % 'true' value for slope coefficient

N=10^4; % Number of observations

% Create data for x, the independent variable
obsxdata=2+0.7*randn(N,1);
% Create data for y, the dependent variable
obsydata=alpha+beta*obsxdata+randn(N,1);

%% Run a quick test, just OLS regression estimation of beta (this is just a double-check that DGP is set up correctly)
[b,bint,r,rint,stats] = regress(obsydata,[ones(N,1),obsxdata]); % Note, no constant term
% b looks fine

%% Initial guess for parameters to be estimated
estparams0=[4.5; 0]; % [alpha; beta] is what we are estimating

%% Set up GMM estimation of linear regression based on moment conditions E[u] E[x*u]=0
W=eye(1);
GMMobjfn=@(estparams) [mean(obsydata-estparams(1)-estparams(2)*obsxdata); mean(obsxdata.*(obsydata-estparams(1)-estparams(2)*obsxdata))]'*W*[mean(obsydata-estparams(1)-estparams(2)*obsxdata); mean(obsxdata.*(obsydata-estparams(1)-estparams(2)*obsxdata))];

[estparams_iter1,fval]=fminsearch(GMMobjfn,estparams0);
% works fine


%% Now, two-iteration efficient GMM
% Use our first-iteration estimate to compute the covar matrix for the moments
OmegaOld=0; % called it old, as do vectorized version below (this is for-loop, gives same answer)
GMMmomentfn=@(y,x) [y-estparams_iter1(1)-estparams_iter1(2)*x; x.*(y-estparams_iter1(1)-estparams_iter1(2)*x)];
E_GMMmomentfn=mean(GMMmomentfn(obsydata',obsxdata'),2);
Omega=zeros(2,2);
for ii=1:N
    Omega=Omega+(GMMmomentfn(obsydata(ii),obsxdata(ii))-E_GMMmomentfn)*(GMMmomentfn(obsydata(ii),obsxdata(ii))-E_GMMmomentfn)'; % Moment condition at each observation
end
Omega=Omega/N;

Wtwoiter=Omega^(-1);

% Now just redefine GMMobjfn using Wtwoiter
GMMobjfn_2nditer=@(estparams) [mean(obsydata-estparams(1)-estparams(2)*obsxdata); mean(obsxdata.*(obsydata-estparams(1)-estparams(2)*obsxdata))]'*Wtwoiter*[mean(obsydata-estparams(1)-estparams(2)*obsxdata); mean(obsxdata.*(obsydata-estparams(1)-estparams(2)*obsxdata))];

[estparams_GMMtwoiter,fval]=fminsearch(GMMobjfn_2nditer,estparams_iter1); % start from first-iter estimate

% Our two-iteration efficient GMM estimate is
estparams_GMMtwoiter

%% NOTE. IT IS NOT POSSIBLE TO COMPUTE Omega WITHOUT A VALUE FOR THE PARAMETER TO BE ESTIMATED!
% That is, cannot just do Efficient GMM, but can do two-iteration efficient GMM

% Note: because we have two moments for two parameters the weighting matrix
% is actually largely unimportant here anyway. So we are more just
% illustrating how we have to use two-iteration efficient GMM (cannot
% estimate Omega until we have completed the first-iteration).




