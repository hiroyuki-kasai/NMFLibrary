% The following vectorized Matlab code implements Algorithm 1 from the
% paper
% Projection onto the probability simplex: An efficient algorithm with a 
% simple proof, and an application, by W. Wang and M.A. Carreira-Perpinan, 
% see https://arxiv.org/abs/1309.1541 
% 
% It projects each column vector in the D N matrix Y onto the probability 
% simplex in D dimensions.

function X = SimplexColProj(Y)

Y = Y'; 
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
X = X'; 