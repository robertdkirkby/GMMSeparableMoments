function out=momentcondnfn_AR1_B(estparams,y_t,sampleMean)
% Output is g (3xT) or G (3x1), when sampleMean=0 or 1, respectively

% estparams=[rho,sigma]
rho=estparams(1);
sigma=estparams(2);

T=length(y_t)-1;

e_t=y_t(2:end)-rho*y_t(1:end-1);

% moment condition functions of each observation
g=zeros(3,T);
g(1,:)=e_t; % E[e_t]=0
g(2,:)=e_t.^2-sigma^2; % E[e_t^2]=sigma^2
g(3,:)=e_t.*y_t(1:end-1); % E[e_t*y_{t-1}]=0

if sampleMean==0
    out=g;
elseif sampleMean==1
    G=mean(g,2); % moment conditions as column vector
    out=G;
end

end