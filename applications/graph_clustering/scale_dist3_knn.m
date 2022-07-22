function A = scale_dist3_knn(D, nn, knn, useSparse)
%SCALE_DIST3_KNN
% A = scale_dist3_knn(D, nn, knn) returns a
% self-tuned affinity matrix A based on the distance
% matrix D. Each observation is only connected to its
% 'knn' neighbors. The affinity values are defined as:
%     A_ii = 0, for all i
%     A_ij = exp(-D_ij / (sigma_i * sigma_j)),
%           if i ~= j and
%           the i-th observation is one of the 'knn'
%           neighbors of the j-th observation
%           or vice versa
%     A_ij = 0, otherwise
% For any i, sigma_i is the Euclidean distance between
% the i-th observation and its nn-th neighbor.
% The returned affinity matrix A is a sparse matrix.
%
% A = scale_dist3_knn(D, nn, knn, useSparse) returns a sparse
% matrix if useSparse is true, and returns a dense matrix otherwise.
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

if (nargin < 4)
    useSparse = true;
end

distSparse = issparse(D);
n = size(D, 1);

if (distSparse)
    max_rows = full(max(sum(D~=0)));
        if (knn > max_rows)
                knn = max_rows;
        end
    max_nonzeros = nnz(D);
    i = zeros(max_nonzeros, 1);
    j = zeros(max_nonzeros, 1);
    sorted_s = zeros(max_nonzeros, 1);
    idx_s = zeros(max_nonzeros, 1);
    current_pos = 0;
    for col_num = 1 : n
        col_nz = D(:, col_num);
        idx_temp = find(col_nz ~= 0);
        col_nz = full(col_nz(col_nz ~= 0));
        col_nnz = length(col_nz);
        i(current_pos+1 : current_pos+col_nnz) = 1 : col_nnz;
        j(current_pos+1 : current_pos+col_nnz) = col_num;
        [sorted, idx_relative] = sort(col_nz);
        sorted_s(current_pos+1 : current_pos+col_nnz) = sorted;
        idx_s(current_pos+1 : current_pos+col_nnz) = idx_temp(idx_relative);
        if (nn > col_nnz)
            ls(col_num) = sorted(end);
        else
            ls(col_num) = sorted(nn);
        end
        current_pos = current_pos + col_nnz;
    end
    ls = sqrt(ls)';
    sorted = sparse(i, j, sorted_s, max_rows, n);
    idx = sparse(i, j, idx_s, max_rows, n);
        j = meshgrid(1:n, 1:knn);
        j = j(:);
        i = full(idx(1:knn, :));
        i = i(:);
    s = full(sorted(1:knn, :));
    s = s(:);
    temp = find(i ~= 0);
    i = i(temp);
    j = j(temp);
    index = [i, j; j, i];
    s = s(temp);
    s = [s; s];
else
    if (nn > n-1)
        nn = n-1;
    end
    if (knn > n-1)
        knn = n-1;
    end
    [sorted, idx] = sort(D);
    ls = sorted(nn+1, :);
    ls = sqrt(ls)';
    j = meshgrid(1:n, 1:knn+1);
    j = j(:);
    i = idx(1:knn+1, :);
    i = i(:);
    I = find(i ~= j);
    i = i(I);
    j = j(I);
    index = [i, j; j, i];
    s = sorted(1:knn+1, :);
    s = s(:);
    s = s(I);
    s = [s; s];
end

A_s = exp( -s ./ (ls(index(:,1)).*ls(index(:,2))) );

if (useSparse)
    [index, i, j] = unique(index, 'rows');
    A_s = A_s(i);
    A = sparse(index(:,1), index(:,2), A_s, n, n);
else
    index = (index(:,2) - 1) * n + index(:, 1);
    A = zeros(n);
    A(index) = A_s;
end
