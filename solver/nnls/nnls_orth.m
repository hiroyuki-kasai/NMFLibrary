% OrthNNLS solves 
% 
% min_{norm2v >=0, V >= 0 and VV^T=D} ||M-U*V||_F^2 
% 
% where the columns of Mn are normalized columns of M 
%
% References:
%       F. Pompili, N. Gillis, P.-A. Absil and F. Glineur, 
%       "Two Algorithms for Orthogonal Nonnegative Matrix Factorization
%       with Application to Clustering," 
%       Neurocomputing 141, pp. 15-25, 2014. 
%
%       This file has been ported from 
%       orthNNLS.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)

function [V,norm2v] = nnls_orth(M,U,Mn) 

    if nargin <= 2 || isempty(Mn) 
        norm2m = sqrt(sum(M.^2,1)); 
        Mn = M.*repmat(1./(norm2m+1e-16),size(M,1),1);  
    end
    [m,n] = size(Mn); 
    [m,r] = size(U); 
    % Normalize columns of U
    norm2u = sqrt(sum(U.^2,1));
    Un = U.*repmat(1./(norm2u+1e-16),m,1);
    A = Mn'*Un; % n by r matrix of angles between the columns of U and M
    [a,b] = max(A'); % best column of U to approx. each column of M
    V = zeros(r,n); 
    % Assign the optimal weights to V(b(i),i) > 0
    for i = 1 : n
        V(b(i),i) = M(:,i)'*U(:,b(i))/norm(U(:,b(i)))^2;
end