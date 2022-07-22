function A = inner_product_knn(D, Xnorm, knn, useSparse)
%INNER_PRODUCT_KNN
% A = inner_product_knn(D, Xnorm, knn) returns an
% affinity matrix A with inner product similarities.
% Each observation is only connected to its 'knn'
% neighbors. The affinity values are defined as:
%     A_ii = 0, for all i
%     A_ij = Xnorm(:,i)' * Xnorm(:,j),
%           if i ~= j and
%           the i-th observation is one of the 'knn'
%           neighbors of the j-th observation
%           or vice versa
%     A_ij = 0, otherwise
% D is used to determine the nearest neighbors, while
% Xnorm is used to compute the affinity values. Also
% note that each column of 'Xnorm' is the vector
% representation of one observation.
% The returned affinity matrix A is a sparse matrix.
%
% A = inner_product_knn(D, Xnorm, knn, useSparse) returns a sparse
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
        idx_s(current_pos+1 : current_pos+col_nnz) = idx_temp(idx_relative);
        current_pos = current_pos + col_nnz;
    end
    idx = sparse(i, j, idx_s, max_rows, n);
        j = meshgrid(1:n, 1:knn);
        j = j(:);
        i = full(idx(1:knn, :));
        i = i(:);
    temp = find(i ~= 0);
    i = i(temp);
    j = j(temp);
    index = [i, j; j, i];
else
    if (knn > n-1)
        knn = n-1;
    end
    [sorted, idx] = sort(D);
    j = meshgrid(1:n, 1:knn+1);
    j = j(:);
    i = idx(1:knn+1, :);
    i = i(:);
    I = find(i ~= j);
    i = i(I);
    j = j(I);
    index = [i, j; j, i];
end

if (useSparse)
    [index, i, j] = unique(index, 'rows');
    A = zeros(size(index, 1), 1);
    for nnz_num = 1 : size(index, 1)
        A(nnz_num) = Xnorm(:, index(nnz_num, 1))' * Xnorm(:, index(nnz_num, 2));
    end
    A = sparse(index(:,1), index(:,2), A, n, n);
else
    A = Xnorm' * Xnorm;
    index = (index(:,2) - 1) * n + index(:, 1);
    A = zeros(n);
    A(index) = A(index);
end
