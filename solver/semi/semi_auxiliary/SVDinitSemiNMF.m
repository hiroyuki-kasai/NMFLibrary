function [U,V] = SVDinitSemiNMF(M,r,init); 
% [U,V] = SVDinitSemiNMF(M,r,init); 
%
% SVD-based initialization for semi-NMF
%
% It guarantees the semi-NMF (U,V) to achieve an error as good as the
% best rank-(r-1) unconstrained approximation, that is, 
%
%         ||M-UV||_F     <=     min_{X, rank(X) <= r-1} ||M-X||_F , 
%
% See Theorem 1 and Algorithm 2 in 
% N. Gillis, A. Kumar, Exact and Heuristic Algorithms for Semi-Nonnegative 
% Matrix Factorization, arXiv, 2014
% 
%
% ****** Input ******
%   M    : m-by-n matrix 
%   r    : factorization rank
%   init : different ways to flip the signs of the factors of the truncated
%           SVD. 
%           Default = 1 (described in the paper), 
%           Otherwise: like in the paper 'Resolving the sign ambiguity in 
%           the singular value decomposition', Bro, Acar and Kolda, 
%           Journal of Chemometrics 22(2):135-140, 2008. 
%
% ****** Output ******
%  (U,V) : an m-by-r matrix U and an r-by-n nonnegative matrix V such that 
%             ||M-UV||_F <= min_{X, rank(X) <= r-1} ||M-X||_F

if nargin <= 2
    init = 1; 
end

if r == 1
    [A,S,B] = svds(M,r); B = B'; 
    if sum(B > 0) < sum(B < 0), B = -B; end
    V = max(B,0);
    U = M/V; 
else
    [A,S,B] = svds(M,r-1); A = A*S; B = B'; 
    [m,n] = size(M); 
    if init == 1 % flip sign to maximize minimum entry
        for i = 1 : r-1
            if min(B(i,:)) < min(-B(i,:))
                B(i,:) = -B(i,:); 
                A(:,i) = -A(:,i);
            end
        end
    else % Using Bro et al. sign flip: 
        loads{1} = A; 
        loads{2} = B'; 
        [sgns,loads] = sign_flip(loads,M); 
        A = loads{1}; 
        B = loads{2}'; 
    end
    if r == 2
       U = [A -A];  
    else
        U = [A -sum(A')']; 
    end
    V = [B; zeros(1,n)]; 
    if r >= 3
        V = V - ones(r,1)* min(0, min(B)); 
    else
        V = V - ones(r,1)*min(B,0); 
    end
end