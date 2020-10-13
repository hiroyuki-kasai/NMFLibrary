% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 24, 2018
% Modified by H.Kasai on July 30, 2018

clc;
clear;
close all;

%% generate synthetic data of (mxn) matrix
m= 150;
n = 150;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 5;


%% Initialize of W and H
options = [];
[x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
options.x_init = x_init;
options.verbose = 2;
options.max_epoch = 1000;
sparse_coeff = 0.5;



%% perform factroization
% NMF-MU
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);


% Sparse-MU (EUC)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
[w_sparse_mu, infos_sparse_mu] = nmf_sparse_mu(V, rank, options);

% Sparse-MU (KL)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
options.metric = 'KL';
[w_sparse_mu_kl, infos_sparse_mu_kl] = nmf_sparse_mu(V, rank, options); 

% Sparse-MU-V (KL)
options.lambda = sparse_coeff;
options.myeps = 1e-20;
options.metric = 'KL';
[w_sparse_mu_kl_v, infos_sparse_mu_kl_v] = nmf_sparse_mu_v(V, rank, options);     


% nsNMF
options.theta = 0.6; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
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

%%

params = struct;

% Objective function
params.cf = 'kl';   %  'is', 'kl', 'ed'; takes precedence over setting the beta value
  % alternately define: params.beta = 1;
params.sparsity = 5;

% Stopping criteria
params.max_iter = 100;
params.conv_eps = 1e-3;
% Display evolution of objective function
params.display   = 0;

% Random seed: any value over than 0 sets the seed to that value
params.random_seed = 1;

% Optional initial values for W 
%params.init_w
% Number of components: if init_w is set and r larger than the number of
% basis functions in init_w, the extra columns are randomly generated
params.r = rank;
% Optional initial values for H: if not set, randomly generated 
%params.init_h

% List of dimensions to update: if not set, update all dimensions.
%params.w_update_ind = true(r,1); % set to false(r,1) for supervised NMF
%params.h_update_ind = true(r,1);

[w, h, objective] = snmf(V, params);


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




