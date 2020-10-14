function [Vlp, yopt, epsiopt] = LPinitSemiNMF(M,r,algolp); 
% [Vlp, yopt, epsiopt] = LPinitSemiNMF(M,r,algolp); 
%
% Initialization for semi-NMF based on the SVD and linear programming
%
% It guarantees the generated factor V to achieve an error as good as the
% best rank-r unconstrained approximation, that is, 
%
%     min_U ||M-UV||_F     <=     min_{X, rank(X) <= r} ||M-X||_F, 
%
% GIVEN THAT the best rank-r unconstrained approximation contains a 
% positive vector in its row space. Otherwise, there is no optimality 
% guarantee. 
%
% See Corollary 4 and Algorithm 3 in 
% N. Gillis, A. Kumar, Exact and Heuristic Algorithms for Semi-Nonnegative 
% Matrix Factorization, arXiv, 2014
% 
%
% ****** Input ******
%   M     : m-by-n matrix 
%   r     : factorization rank
%   algolp: sovler used to solve the linear systems
%           algolp = 1: linprog of Matlab (default)
%           algolp = 2: CVX (see http://cvxr.com/)
%
% ****** Output ******
%   Vlp : an r-by-n nonnegative matrix such that U*Vlp approx M for some U. 

if nargin <= 2
    algolp = 1;
end
[U,S,V] = svds(M,r); 
V = V';
% Remove zero columns
I = sum(abs(V)); 
I = find(I > 1e-6*mean(I)); 
% Only keep column whose l_1 norm is larger than 1e-6 * average
V = V(:,I); 

[r, n] = size(V);
% Sign flip 
for k = 1 : r
    if abs(min(V(k,:))) > max(V(k,:))
        V(k,:) = -V(k,:);
    end
end
% 1. Test if epsi = 0 works, that is, is -V'*yopt <= -1 feasible? 
[y,eflag] = linsys_semiNMF(V,0,algolp); 
if eflag == 1
    epsiopt = 0;
    yopt = y; 
else % Bisection method
    epsimin = 0; % Lower bound for epsilon
    epsimax = max(abs(V(:))); % Upper bound for epsilon
    epsiopt = epsimax; 
    epsimax0 = epsimax; 
    epsi = (epsimin+epsimax)/2;  
    while (epsimax - epsimin)/epsimax0 > 1e-3 || eflag == -1
        [y,eflag] = linsys_semiNMF(V,epsi,algolp); 
        % is -(V+epsi)'*y >= 1 feasible? 
        if eflag == -1
            epsimin = epsi;
        else 
            epsimax = epsi;
            epsiopt = epsi; 
            yopt = y; 
        end
        epsi = (epsimin+epsimax)/2; 
    end 
end
x = (V+epsiopt)'*yopt;
for i = 1 : r
    alpha(i,1) = max(0, max(-V(i,:)'./x)); 
end
V = max(0, V + alpha*x'); 
% The max with 0 is not strictly necessary but avoids numerical errors 
% that would lead to small negative entries in V

% Re-add zero columns
[m,n] = size(M); 
Vlp = zeros(r,n); 
Vlp(:,I) = V; 