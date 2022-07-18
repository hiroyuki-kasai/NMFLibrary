function [ gamma ] = gamma_beta( beta )
% Computes the exponennt \gamma(\beta) from 
% "Algorithms for nonnegative matrix factorization with the beta-"divergence"

if beta<1
    gamma = 1/(2-beta);
    
elseif beta>2
    gamma = 1/(beta-1);
    
else
    gamma = 1;

end

