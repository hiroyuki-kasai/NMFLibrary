function x = SimplexProj(y)

% Given y,  computes its projection x* onto the set 
% 
%       S = { x | x >= 0 and sum(x) <= 1 }, 
% 
% that is, x* = argmin_x ||x-y||_2  such that x in S. 
% 
% If y is a matrix, is projects its columns onto S to obtain x. 
%  
%
% x = SimplexProj(y)
%
% ****** Input ******
% y    : input vector.
%
% ****** Output ******
% x    : projection of y onto Delta.

x = max(y,0); 
if size(x,1) == 1
    x = min(x,1); 
else
    K = find(sum(x) > 1); 
    x(:,K) = SimplexColProj(y(:,K));
end
end