function [idx, iter, obj, H] = symnmf_cluster(X, k, options)
%SYMNMF_CLUSTER
% [idx, iter, obj, H] = symnmf_cluster(X, k, options)
%
% This function performs graph clustering on a data matrix.
%
% Input:
% X - NxP data matrix (N observations with dimension P),
%     where each row is one observation,
%     and each column is one feature/variable.
%     (If an NxN similarity matrix is available,
%      please call 'symnmf_anls' directly.)
% k - Specifying the number of clusters.
% options - A structure of optional parameters for
%           clustering (see details below).
%
% Default configurations:
%   options.graph_type = 'sparse';
%   options.similarity_type = 'gaussian';
%   options.graph_objfun = 'ncut';
%   (no options.kk given)
%   options.nn = 7;
%   options.tol = 1e-3;
%   options.maxiter = 10000;
%   options.rep = 1;
%   (no options.Hinit given)
%   options.computeobj = true;
%   options.alg = 'anls';
%
% Output:
% idx - Nx1 vector, containing the clustering assignment
%       of each observation.
% iter - Number of iterations actually used
% obj - Final value of the objective function
%           f(H) = ||A - HH'||_F^2
% H - Final result of low-rank matrix H
%
% Available options:
% 1. options.graph_type
%    = 'full': to construct a fully-connected graph
%    = 'sparse': to construct a sparse graph where
%                each observation is connected to
%                its KK nearest neighbors. (DEFAULT)
%
%    (Note: Only undirected graph is supported,
%           so the resulting similarity matrix
%           is symmetric.)
%
% 2. options.similarity_type
%    = 'gaussian': to use self-tuning Gaussian similarity,
%                  proposed in [Zelnik-Manor and Perona,
%                  2004] (see more details in functions
%                  'scale_dist3' and 'scale_dist3_knn') (DEFAULT)
%    = 'inner_product': to use inner product similarity
%                       (good choice for text data)
%
% 3. options.graph_objfun
%    = 'ncut': to use normalized cut (DEFAULT)
%    = 'rcut': to use ratio cut
%
% 4. options.kk (DEFAULT is unset, i.e. depending on N)
%    specifies the number of nearest neighbors WHEN
%    options.graph_type = 'sparse'.
%
%    The default value depends on N:
%    option.kk = floor(log2(N)) + 1;
%
% 5. options.nn (DEFAULT is 7)
%    specifies the nn-th neighbor, which is used in
%    the self-tuning Gaussian similarity WHEN
%    options.similarity_type = 'gaussian'.
%
% 6. options.tol (DEFAULT is 1e-3)
%    controls the termination of SymNMF algorithm.
%
% 7. options.maxiter (DEFAULT is 10000)
%    limits the number of iteration allowed in each run.
%
% 8. options.rep (DEFAULT is 1)
%    sets the nubmer of runs of SymNMF algorithm.
%
%    The return values 'idx', 'iter', 'obj', and 'H' are
%    corresponding to a single run, which is the run that
%    yields the lowest objective function value.
%
%    Using multiple runs of SymNMF may help avoid local minima.
%
% 9. options.Hinit (DEFAULT is unset, i.e. random initialization)
%    sets the initialization of matrix H in each run.
%
%    options.Hinit is a 3-dim array. The 3rd dimension may imply
%    the choice of options.rep, and each 2-dim array in the 1st
%    and 2nd dimensions is a NxK nonnegative matrix.
%
%    Random initialization is recommended. If options.Hinit has to
%    be a fixed set of matrices for testing, the instructions on
%    how to initialize H should be followed. Please consult the
%    usage of 'symnmf_newton'.
%
% 10. options.computeobj (DEFAULT is true)
%     specifies whether to compute the objective value
%     f(H) at the final solution H.
%
% 11. options.alg
%     = 'anls' to use ANLS algorithm (DEFAULT)
%     = 'newton' to use Newton-like algorithm
%
% This function is used for experiments in the following paper:
%     Da Kuang, Chris Ding, Haesun Park,
%     Symmetric Nonnegative Matrix Factorization for Graph Clustering,
%     The 12th SIAM International Conference on Data Mining (SDM '12), pp. 106--117.
% Please cite this paper if you find this code useful.
%

[n, p] = size(X);

if ~exist('options', 'var')
    graph_type = 'sparse';
    similarity_type = 'gaussian';
    graph_objfun = 'ncut';
    kk = floor(log2(n)) + 1;
    nn = 7;
    tol = 1e-3;
    maxiter = 10000;
    rep = 1;
    Hinit = [];
    computeobj = true;
    alg = 'anls';
else
    if isfield(options, 'graph_type')
        graph_type_names = {'full', 'sparse'};
        j = find(strcmpi(options.graph_type, graph_type_names));
        if ~isempty(j)
            graph_type = graph_type_names{j};
        else
            error('Invalid options.graph_type value!');
        end
    else
        graph_type = 'sparse';
    end
    if isfield(options, 'similarity_type')
        similarity_type_names = {'gaussian', 'inner_product'};
        j = find(strcmpi(options.similarity_type, similarity_type_names));
        if ~isempty(j)
            similarity_type = similarity_type_names{j};
        else
            error('Invalid options.similarity_type value!');
        end
    else
        similarity_type = 'gaussian';
    end
    if isfield(options, 'graph_objfun')
        graph_objfun_names = {'ncut', 'rcut'};
        j = find(strcmpi(options.graph_objfun, graph_objfun_names));
        if ~isempty(j)
            graph_objfun = graph_objfun_names{j};
        else
            error('Invalid options.graph_objfun value!');
        end
    else
        graph_objfun = 'ncut';
    end
    if isfield(options, 'kk')
        if ~isempty(options.kk) & isnumeric(options.kk) & options.kk < n
            kk = options.kk;
        else
            error('options.kk must be an integer less than N!');
        end
    else
        kk = floor(log2(n)) + 1;
    end
    if isfield(options, 'nn')
        if ~isempty(options.nn) & isnumeric(options.nn) & options.nn < n
            nn = options.nn;
        else
            error('options.nn must be an integer less than N!');
        end
    else
        nn = 7;
    end
    if isfield(options, 'tol')
        if ~isempty(options.tol) & isnumeric(options.tol) & options.tol > 0 & options.tol < 1
            tol = options.tol;
        else
            error('options.tol must be a real number and 0 < options.tol < 1!');
        end
    else
        tol = 1e-3;
    end
    if isfield(options, 'maxiter')
        if ~isempty(options.maxiter) & isnumeric(options.maxiter) & options.maxiter > 0
            maxiter = options.maxiter;
        else
            error('options.maxiter must be a positive integer!');
        end
    else
        maxiter = 10000;
    end
    if isfield(options, 'rep')
        if ~isempty(options.rep) & isnumeric(options.rep) & options.rep > 0
            rep = options.rep;
        else
            error('options.rep must be a positive integer!');
        end
        if isfield(options, 'Hinit') & size(options.Hinit, 3) ~= rep
            error('The third dimension of options.Hinit must match options.rep!')
        end
    else
        if isfield(options, 'Hinit') & ~isempty(options.Hinit)
            rep = size(options.Hinit, 3);
        else
            rep = 1;
        end
    end
    if isfield(options, 'Hinit')
        if ~isempty(options.Hinit) & isnumeric(options.Hinit) & size(options.Hinit, 1) == n & size(options.Hinit, 2) == k
            Hinit = options.Hinit;
        else
            error('The size of each initialization of H must be Nxk!');
        end
    else
        Hinit = [];
    end
    if isfield(options, 'computeobj')
        if ~isempty(options.computeobj) & (isnumeric(options.computeobj) | isboolean(options.computeobj))
            computeobj = options.computeobj;
        else
            error('options.computeobj must be boolean or a real number!');
        end
    else
        computeobj = true;
    end
    if computeobj == false & rep > 1
        error('options.computeobj must be true if options.rep > 1!');
    end
    if isfield(options, 'alg')
        alg_names = {'anls', 'newton'};
        j = find(strcmpi(options.alg, alg_names));
        if ~isempty(j)
            alg = alg_names{j};
        else
            error('Invalid options.alg value!');
        end
    else
        alg = 'anls';
    end
end

D = dist2(X, X);
if strcmp(graph_type, 'full') & strcmp(similarity_type, 'gaussian')
    A = scale_dist3(D, nn);
elseif strcmp(graph_type, 'full') & strcmp(similarity_type, 'inner_product')
    A = X * X';
elseif strcmp(graph_type, 'sparse') & strcmp(similarity_type, 'gaussian')
    A = scale_dist3_knn(D, nn, kk, true);
else % graph_type == 'sparse' & similarity_type == 'inner_product'
    Xnorm = X';
    d = 1./sqrt(sum(Xnorm.^2));
    Xnorm = bsxfun(@times, Xnorm, d);
    A = inner_product_knn(D, Xnorm, knn, true);
    clear Xnorm, d;
end
clear D;

if strcmp(graph_objfun, 'ncut')
    dd = 1 ./ sum(A);
    dd = sqrt(dd);
    A = bsxfun(@times, A, dd);
    A = A';
    A = bsxfun(@times, A, dd);
    clear dd;
end
A = (A + A') / 2;

params.maxiter = maxiter;
params.tol = tol;

obj_best = Inf;
for i = 1 : rep
    if isempty(Hinit)
        if strcmp(alg, 'newton')
            [sol, infos] = symm_newton(A, k, params);
        else % strcmp(alg, 'anls')
            [sol, infos] = symm_anls(A, k, params);
        end
    else
        params.Hinit = Hinit(:, :, i);
        if strcmp(alg, 'newton')
            [sol, infos] = symm_newton(A, k, params);
        else % strcmp(alg, 'anls')
            [sol, infos] = symm_anls(A, k, params);
        end
    end
    H = sol.H;
    [max_val, idx] = max(H, [], 2);
    if infos.cost(end) < obj_best
        idx_best = idx;
        iter_best = infos.iter(end);
        obj_best = infos.cost(end);
        H_best = H;
    end
end

idx = idx_best;
iter = iter_best;
obj = obj_best;
H = H_best;
