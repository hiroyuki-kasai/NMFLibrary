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
m = 500;
n = 500;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 50;


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


% nsNMF (MU)
options.theta = 0.2; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
[w_nsnmf, infos_nsnmf] = ns_nmf(V, rank, options); 

% nsNMF (MU)
options.theta = 0.2; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
options.update_alg = 'apg';
[w_nsnmf_apg, infos_nsnmf_apg] = ns_nmf(V, rank, options);   



%% plot
display_graph('epoch','cost',{'NMF-MU','NS-NMF (MU)','NS-NMF (APG)'}, ...
    {w_nmf_mu, w_nsnmf,w_nsnmf_apg}, {infos_nmf_mu,infos_nsnmf,infos_nsnmf_apg});


% sparseness 
%   defined by
%       Patrik O. Hoyer, 
%       "Non-negative matrix factorization with sparseness constraints," 
%       Journal of Machine Learning Research, vol.5, pp.1457-1469, 2004.

display_sparsity_graph({'NMF-MU','NS-NMF (MU)','NS-NMF (APG)'}, {w_nmf_mu, w_nsnmf,w_nsnmf_apg});




