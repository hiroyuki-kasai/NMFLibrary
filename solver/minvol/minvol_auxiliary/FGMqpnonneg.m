function [W,e] = FGMqpnonneg(A,C,W0,maxiter,proj) 

% Fast gradient method to solve nonnegative least squares.  
% See Nesterov, Introductory Lectures on Convex Optimization: A Basic 
% Course, Kluwer Academic Publisher, 2004. 
% 
% This code solves: 
% 
%     min_{x_i in R^r_+} sum_{i=1}^m ( x_i^T A x_i - 2 c_i^T x_i ),  
% 
% [W,e] = FGMfcnls(A,C,W0,maxiter) 
%
% ****** Input ******
% A      : Hessian for each row of W, positive definite
% C      : linear term <C,W>
% W0     : m-by-r initial matrix
% maxiter: maximum numbre of iterations (default = 500). 
% proj   : =1, nonnegative orthant
%          =2, nonnegative orthant + sum-to-one constraints on columns
%
% ****** Output ******
% W      : approximate solution of the problem stated above. 
% e      : e(i) = error at the ith iteration

[m,r] = size(C); 
% Initialization of W  
if nargin <= 2 || isempty(W0)
    W0 = zeros(m,r); 
end
if nargin <= 3
    maxiter = 500; 
end
if nargin <= 4
    proj = 1; 
end
% Lipschitz constant 
L = norm(A,2); 
% Extrapolation parameter 
condAm1 = 1/cond(A); %  smallest / largest  singular value of A 
beta = (1-sqrt(condAm1)) / (1 + sqrt(condAm1)); 
% Project initialization onto the feasible set
if proj == 1
    W = max(W0,0);
elseif proj == 2
    W = SimplexColProj( W0 );
end
Y = W; % second sequence
i = 1;
% Stop if ||W^{k}-W^{k+1}||_F <= delta * ||W^{0}-W^{1}||_F
delta = 1e-6;
eps0 = 0; eps = 1;  
while i <= maxiter && eps >= delta*eps0
    % Previous iterate
    Wp = W; 
%     % FGM Coefficients  
%     %alpha(i+1) = ( sqrt(alpha(i)^4 + 4*alpha(i)^2 ) - alpha(i)^2) / (2); 
%     %beta(i) = alpha(i)*(1-alpha(i))/(alpha(i)^2+alpha(i+1)); 
    % Projected gradient step from Y
    W = Y - (Y*A-C) / L ; 
    % Projection 
    if proj == 1
        W = max(W,0);
    elseif proj == 2
        W = SimplexColProj( W );
    end
    % `Optimal' linear combination of iterates
    Y = W + beta*(W-Wp); 
    % Error 
    e(i) = sum(sum((W'*W).*A)) - 2*sum(sum(W.*(C))); 
    % Restart: fast gradient methods do not guarantee the objective
    % function to decrease, a good heursitic seems to restart whenever it
    % increases although the global convergence rate is lost! This could
    % be commented out. 
    if i >= 2 && e(i) > e(i-1)
        Y = W; 
    end 
    if i == 1
        eps0 = norm(W-Wp,'fro'); 
    end
    eps = norm(W-Wp,'fro'); 
    i = i + 1; 
end  