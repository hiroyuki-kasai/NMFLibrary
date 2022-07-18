% Default initialization for H in the NNLS problem 
% 
%         min_{H >= 0} ||X - WH||_F^2 
%
% It uses orthogonal NMF (that is, HH^T=I) of W is ill conditioned
% otherwuse it uses the projection of the unconstrained least-squares
% solution. 

function H = nnls_init_nmflibrary(X,W,WtW,WtX)

if cond(W) > 1e6 % Assign each column of X to the closest column of W
    % in terms of angle: this is the optimal solution with
    % V having a single non-zero per column.
    H = nnls_orth(X,W);
else  % Projected LS solution + scaling
    if issparse(X)
        H = max(0, pinv(W)*X);
    else
        H = max(0, W\X);
    end 
    % Scale
    alpha = sum(sum( H.*(WtX) ) ) / sum(sum( WtW.*(H*H')));
    H = H*alpha;
end
% Check that no rows of H is zeros 
% If it is the case, set them to small random numbers
zerow = find(sum(H') == 0); 
H(zerow, :) = 0.001*max(H(:))*rand(length(zerow),size(H,2)); 
