% Solve with fast grad
% 
% min_{W >= 0} || X-WH ||_F^2 
% such that 
% sparsity(W) = s 

function [W,XHt,HHt] = fastgradsparseNNLS(X,H,W,options) 

if ~isfield(options,'delta')
    options.delta = 0.1; 
end
if ~isfield(options,'inneriter')
    options.inneriter = 100; 
end

XHt = X*H'; 
HHt = H*H'; 

i = 1; 
alpha0 = 0.1; 
alpha(1) = alpha0;
Yw = W; 
L = norm(HHt,2); 
Wn = W; 
normWWn0 = -1; 

while i <= options.inneriter && norm(Wn-W) >= options.delta*normWWn0
    W = Wn; 
    gradYw =  Yw*HHt - XHt; 
    Wn = max(0, Yw - 1/L * gradYw); 
    if ~isempty(options.s)
        if options.colproj == 0
            Wn = weightedgroupedsparseproj_col( Wn , options.s, options );  
        else
            for k = 1 : size(Wn,2) 
                Wn(:,k) = weightedgroupedsparseproj_col( Wn(:,k) , options.s, options );  
            end
        end
    end
    alpha(i+1) = ( sqrt(alpha(i)^4 + 4*alpha(i)^2 ) - alpha(i)^2) / (2);
    beta(i) = alpha(i)*(1-alpha(i))/(alpha(i)^2+alpha(i+1));
    Yw = Wn + beta(i)*(Wn-W);
    if i == 1
        normWWn0 = norm(W-Wn,'fro'); 
    end
    i = i+1; 
end