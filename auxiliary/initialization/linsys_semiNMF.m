function [y,eflag] = linsys_semiNMF(V,epsi,algolp); 
% [y,eflag] = linsys_semiNMF(V,epsi,algolp); 
%
% Check whether the following linear system: (V+epsi)'*y >= 1 is feasible. 
% 
% If it is feasible, y is a solution and eflag = 1. 
% Otherwize, eflag = -1. 
% 
% algolp == 1 : uses linprog of Matlab (default)
% else        : uses CVX available at http://cvxr.com/

if nargin <= 2
    algolp = 1;
end
[r,n] = size(V); 
% ************************ Using linprog  ************************
if algolp == 1
    OPTIONS.Display = 'off';  
    [y, ~, eflag] = linprog(zeros(r,1),-(V+epsi)',-ones(n,1),[],[],[],[],[],OPTIONS); 
    if eflag ~= 1
        eflag = -1;
    end
else
% ************************ Using CVX  ************************
    cvx_begin quiet
    variable y(r);
        minimize 0 
        (V+epsi)'*y >= 1;
    cvx_end
    if sum(isnan(y)) == 0
        eflag = 1;
    else
        eflag = -1;
    end
end