function out=momentcondnfn_stochvol(estparams,delta_r_t,r_t_lag,sampleMean)
% Output is g (4xT) or G (4x1), when sampleMean=0 or 1, respectively

% estparams=[alpha,beta,sigma,gamma]
alpha=estparams(1);
beta=estparams(2);
sigma=estparams(3);
gamma=estparams(4);

T=length(delta_r_t);

% u_t=r_t-r_{t-1}-alpha-beta*r_{t-1}
u_t=delta_r_t-alpha-beta*r_t_lag;

% moment condition functions of each observation
g=zeros(4,T);
g(1,:)=u_t; % u_t-0
g(2,:)=(u_t.^2)-((sigma^2)*(r_t_lag.^(2*gamma)));
g(3,:)=u_t.*r_t_lag;
g(4,:)=(u_t.^2-(sigma^2)*(r_t_lag.^(2*gamma))).*r_t_lag;

if sampleMean==0
    out=g;
elseif sampleMean==1
    G=mean(g,2); % moment conditions as column vector
    out=G;
end

end