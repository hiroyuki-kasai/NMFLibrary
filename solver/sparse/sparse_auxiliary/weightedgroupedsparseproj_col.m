% Function weightedgroupedsparseproj applied to the columns of a matrix,
% with corresponding weights W 

function [Xp,numiter] = weightedgroupedsparseproj_col(X,s,options)

if nargin <= 2
    options = [];
end
[m,r] = size(X); 
for i = 1 : r
    x{i} = X(:,i); 
end
[xp,gxpmu,numiter,newmu] = weightedgroupedsparseproj(x,s,options); 
for i = 1 : r
    Xp(:,i) = xp{i}; 
end