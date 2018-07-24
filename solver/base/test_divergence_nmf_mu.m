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
% 'EUC';
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);


[W,H,errs,vout] = nmf_kl(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
          'W0', x_init.W, 'H0', x_init.H, 'myeps', 1e-16);    

% 'KL';
options.metric = 'KL';
[w_nmf_mu_kl, infos_nmf_mu_kl] = nmf_mu(V, rank, options);

%return;

[W,H,errs,vout] = nmf_amari(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
           'W0', x_init.W, 'H0', x_init.H, 'myeps', 1e-16);  

% 'ALPHA-D';
options.metric = 'ALPHA-D';
[w_nmf_mu_alpha, infos_nmf_mu_alpha] = nmf_mu(V, rank, options);

%return;

[W,H,errs,vout] = nmf_beta(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
           'W0', x_init.W, 'H0', x_init.H, 'myeps', 1e-16);  

% 'BETA-D';
options.metric = 'BETA-D';
[w_nmf_mu_beta, infos_nmf_mu_beta] = nmf_mu(V, rank, options);




%% plot
display_graph('epoch','cost',{'NMF-MU (EUC)','NMF-MU (KL)','NMF-MU (ALPHA)','NMF-MU (BETA)'}, ...
    {w_nmf_mu, w_nmf_mu_kl,w_nmf_mu_alpha,w_nmf_mu_beta}, ...
    {infos_nmf_mu,infos_nmf_mu_kl,infos_nmf_mu_alpha,infos_nmf_mu_beta});


