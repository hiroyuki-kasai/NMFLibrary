function [seminnrank,U,V] = seminonnegativerank(M); 
% [seminnrank,U,V] = seminonnegativerank(M); 
%
% Computes the semi-nonnegative rank and a corresponding factorization of
% matrix M: M = UV, where U has r columns and V >= 0 has r rows and 
% r = semi-nonnegative rank of M. 
%
% See Corollary 3 in 
% N. Gillis, A. Kumar, Exact and Heuristic Algorithms for Semi-Nonnegative 
% Matrix Factorization, arXiv, 2014
% 
% ****** Input ******
%   M     : m-by-n matrix 
%
% ****** Output ******
%   seminnrank  : the (numerical) semi-nonnegative rank of M
%   (U,V)       : M = UV, V >= 0 and U (resp. V) has 'seminnrank' columns
%                  (resp. rows)

r = rank(M); 
if issparse(M) ~= 1
    [U,S,V] = svd(M); 
    V = V(:,1:r)'; 
else
    [U,S,V] = svds(M,r); 
    V = V'; 
end

[y,eflag] = linsys_semiNMF(V,0); 
if eflag == 1
    seminnrank = r; 
    if nargout >= 2
        V = LPinitSemiNMF(M,r); 
        U = M/V; 
    end
else
    seminnrank = r + 1; 
    if nargout >= 2
        [U,V] = SVDinitSemiNMF(M,r+1); 
    end
end