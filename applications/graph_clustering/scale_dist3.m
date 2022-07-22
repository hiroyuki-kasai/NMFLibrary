function A = scale_dist3(D, nn)
%SCALE_DIST3
% A = scale_dist3(D, nn) returns a self-tuned affinity
% matrix A based on the distance matrix D. The affinity
% values are defined as:
%     A_ij = exp(-D_ij / (sigma_i * sigma_j)), for i ~= j
%     A_ii = 0,                                for all i
% For any i, sigma_i is the Euclidean distance between
% the i-th observation and its nn-th neighbor.
% The returned affinity matrix A is a dense matrix.
%
% Assumptions on the distance matrix D:
%   When D is a dense matrix:
%     D_ij is the squared Euclidean distance between the
%     i-th and j-th observations.
%   When D is a sparse matrix:
%   (e.g. constructed for image segmentation)
%     If D_ij is nonzero, it is the squared Euclidean
%       distance between the i-th and j-th observations.
%     If D_ij (i~=j) is zero, it means the distance
%       between the i-th and j-th observations is infinity
%       (i.e. the corresponding affinity value is 0).
%     Finally, D_ii=0 (for all i) by definition.
%
% This method was proposed in the following paper:
%     L. Zelnik-Manor, P. Perona,
%     Self-tuning spectral clustering.
%     Advances in Neural Information Processing Systems 17 (NIPS '04), pp. 1601--1608.
% The authors also posted their Matlab code at:
%     http://webee.technion.ac.il/~lihi/Demos/SelfTuningClustering.html
% However, their implementation is different from the
% definition of A in the paper. In particular,
%     A_ij = exp(-D_ij / max((sigma_i*sigma_j), 0.004))
% in their 'scale_dist' function.
%
% Our 'scale_dist3' function here implements the
% original definition of A as stated in the beginning
% of this help document.
%
% This function is used for experiments in the following paper:
%     Da Kuang, Chris Ding, Haesun Park,
%     Symmetric Nonnegative Matrix Factorization for Graph Clustering,
%     The 12th SIAM International Conference on Data Mining (SDM '12), pp. 106--117.
% Please cite this paper if you find this code useful.
%

distSparse = issparse(D);
n = size(D, 1);

if (distSparse)
    for i = 1 : n
        col_nz = D(:, i);
        col_nz = col_nz(col_nz ~= 0);
        [sorted, idx] = sort(col_nz);
        if (nn > length(col_nz))
            ls(i) = sorted(end);
        else
            ls(i) = sorted(nn);
        end
    end
    ls = sqrt(ls)';
    [i, j, s] = find(D);
    A_s = exp( -s ./ (ls(i).*ls(j)) );
    A = sparse(i, j, A_s, n, n);
else
    if (nn > n-1)
        nn = n-1;
    end
    [sorted, idx] = sort(D);
    ls = sorted(nn+1, :);
    ls = sqrt(ls);
    A = exp(-D./(ls'*ls));
        ZERO_DIAG = ~eye(n);
        A = A .* ZERO_DIAG;
end
