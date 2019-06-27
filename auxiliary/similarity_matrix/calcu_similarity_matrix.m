function [ A ] = calcu_similarity_matrix( X );

% X, data matrix, each ROW contains one signal

[n, p] = size(X);
nn = 7; kk = floor(log2(n)) + 1; %default value
D = dist2(X, X); %distance matrix

A = scale_dist3_knn(D, nn, kk, true);
dd = 1 ./ sum(A);
dd = sqrt(dd);
A = bsxfun(@times, A, dd);
A = A';
A = bsxfun(@times, A, dd);

A = (A + A') / 2;

end

