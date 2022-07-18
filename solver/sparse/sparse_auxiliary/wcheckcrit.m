% Checks whetehr the largest entries of x are equal to one another 

function [xcrit,maxx] = wcheckcrit(x,w,precision); 

if nargin <= 2
    precision = 1e-6;
end

indi = find(w > 0); 
xi = x(indi); 
wi = w(indi); 

xiwi = xi./wi; 

maxx = max( xiwi ); 

indi = find(abs(xiwi - maxx) < precision); 
if length(indi) > 1   
    xcrit = maxx;
else
    xcrit = [];
end