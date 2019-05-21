% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 24, 2017

clc;
clear;
close all;

%% generate synthetic data of (mxn) matrix       
m = 150;
n = 150;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 5;


%% Initialize of W and H
options = [];
[x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
options.x_init = x_init;
options.verbose = 2;
options.max_epoch = 100;
sparse_coeff = 0.5;



%% perform factroization
% NMF-MU
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);


%     [W,H,errs,vout] = nmf_euc_sparse_es(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
%          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff);     

% Sparse-MU (EUC)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
[w_sparse_mu, infos_sparse_mu] = nmf_sparse_mu(V, rank, options);


%     [W,H,errs,vout] = nmf_kl_sparse_es(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
%          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff); 

% Sparse-MU (KL)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
options.metric = 'KL';
[w_sparse_mu_kl, infos_sparse_mu_kl] = nmf_sparse_mu(V, rank, options); 


%     [W,H,errs,vout] = nmf_kl_sparse_v(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
%          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff);     


% Sparse-MU-V (KL)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
options.metric = 'KL';
[w_sparse_mu_kl_v, infos_sparse_mu_kl_v] = nmf_sparse_mu_v(V, rank, options);     



% nsNMF
options.theta = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
[w_nsnmf, infos_nsnmf] = ns_nmf(V, rank, options); 

% SparseNMF
options.lambda = 1000; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
[w_sparsenmf, infos_sparsenmf] = sparse_nmf(V, rank, options);  

% NMF with sparse constrained
options.lambda = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
[w_nmfsc, infos_nmfsc] = nmf_sc(V, rank, options);       



%% plot
display_graph('epoch','cost',{'NMF-MU','Sparse-MU','Sparse-MU-KL','Sparse-MU-V','NS-NMF','Sparse-NMF','NMF-SC'}, ...
    {w_nmf_mu, w_sparse_mu,w_sparse_mu_kl,w_sparse_mu_kl_v,w_nsnmf,w_sparsenmf,w_nmfsc}, ...
    {infos_nmf_mu,infos_sparse_mu,infos_sparse_mu_kl,infos_sparse_mu_kl_v,infos_nsnmf,infos_sparsenmf,infos_nmfsc});


% sparseness 
%   defined by
%       Patrik O. Hoyer, 
%       "Non-negative matrix factorization with sparseness constraints," 
%       Journal of Machine Learning Research, vol.5, pp.1457-1469, 2004.

display_sparsity_graph({'NMF-MU','Sparse-MU','Sparse-MU-KL','Sparse-MU-V','NS-NMF','Sparse-NMF','NMF-SC'},...
    {w_nmf_mu, w_sparse_mu,w_sparse_mu_kl,w_sparse_mu_kl_v,w_nsnmf,w_sparsenmf,w_nmfsc});




