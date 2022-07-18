% calculate non-negative leaset squares
function H = calc_nls_nmf(X, W, lambda)
    R = size(W,2);
    %H = inv(W'*W + lambda * eye(R)) * W'* X;
    H = (W'*W + lambda * eye(R)) \ W'* X;
    H = max(H, 1e-16);
end

