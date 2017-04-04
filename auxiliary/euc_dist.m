function [ dist ] = euc_dist(X, Y)

    dist = sum(sum((X-Y).^2));
    
end

