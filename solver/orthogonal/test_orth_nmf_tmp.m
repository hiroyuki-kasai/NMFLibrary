% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 23, 2017

clc;
clear;
close all;

%% generate synthetic data of (mxn) matrix       
m = 1000;
n = 100;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 25;


%% Initialize of W and H
options = [];
[x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
options.x_init = x_init;
options.verbose = 2;
options.max_epoch = 300;



%% perform factroization
% NMF-MU
%[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);

% Orth-MU for H
options.orth_h    = 1;
options.norm_h    = 2;
options.orth_w    = 0;
options.norm_w    = 0;    
[w_orth_h_nmf_mu, infos_orth_h_nmf_mu] = nmf_orth_mu(V, rank, options);

% Orth-MU for W
options.orth_h    = 0;
options.norm_h    = 0;
options.orth_w    = 1;
options.norm_w    = 2; 
[w_orth_w_nmf_mu, infos_orth_w_nmf_mu] = nmf_orth_mu(V, rank, options);


opts.reps = 1;                    % the number of initializations
opts.itrMax = options.max_epoch;    % the maximum number of updates
opts.wo = 1; 
[nmf_so.w, nmf_so.h, obj_best, orth_best] = nmf_so_org(V, rank, opts);
infos_nmf_so.cost = obj_best;
infos_nmf_so.orth = orth_best;
infos_nmf_so.epoch = 1:length(obj_best);

options.wo = 1;
[nmf_hals_so, infos_nmf_hals_so] = nmf_hals_so(V, rank, options);

%[W,H,res,iter,REC]=weakorthonmf(V,x_init.W,x_init.H,rank,0,'verbose',2, 'max_iter',5000);


%[w_onpmf.W, w_onpmf.H, relError, actualIters] = onpmf(V, rank, options.max_epoch);


%% plot
%display_graph('epoch','cost', {'NMF-MU', 'Orth-MU'}, {w_nmf_mu, w_orth_nmf_mu}, {infos_nmf_mu, infos_orth_nmf_mu});
%display_graph('epoch','orth', {'NMF-MU','Orth-MU'}, {w_nmf_mu, w_orth_nmf_mu}, {infos_nmf_mu, infos_orth_nmf_mu});
display_graph('epoch','cost', {'Orth-MU-H', 'Orth-MU-W', 'NMF-SO', 'NMF-HALS-SO'}, {w_orth_h_nmf_mu, w_orth_w_nmf_mu, nmf_so, nmf_hals_so}, {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_so, infos_nmf_hals_so});
display_graph('epoch','orth', {'Orth-MU-H', 'Orth-MU-W', 'NMF-SO', 'NMF-HALS-SO'}, {w_orth_h_nmf_mu, w_orth_w_nmf_mu, nmf_so, nmf_hals_so}, {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_so, infos_nmf_hals_so});



