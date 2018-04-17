function lambda = g_lambda(L, mu)
%G_LAMBDA Computes lambda
%   Input: L, mu
%   Output: lambda

q = mu/L;

if (1/q > 10^6)
    lambda = 10 * mu;
elseif (1/q > 10^4)
    lambda = mu;
else
    lambda = mu/10;
end

end

