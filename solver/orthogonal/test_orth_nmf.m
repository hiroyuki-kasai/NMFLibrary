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
m = 100;
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
options.norm_h    = 1;
options.orth_w    = 0;
options.norm_w    = 0;    
[w_orth_h_nmf_mu, infos_orth_h_nmf_mu] = nmf_orth_mu(V, rank, options);

% Orth-MU for W
options.orth_h    = 0;
options.norm_h    = 0;
options.orth_w    = 1;
options.norm_w    = 1; 
[w_orth_w_nmf_mu, infos_orth_w_nmf_mu] = nmf_orth_mu(V, rank, options);

% NMF-HALS-SO
options.wo = 1;
[nmf_hals_so, infos_nmf_hals_so] = nmf_hals_so(V, rank, options);




%% plot
display_graph('epoch','cost', {'Orth-MU-H', 'Orth-MU-W', 'NMF-HALS-SO'}, {w_orth_h_nmf_mu, w_orth_w_nmf_mu, nmf_hals_so}, {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_hals_so});
display_graph('epoch','orth', {'Orth-MU-H', 'Orth-MU-W', 'NMF-HALS-SO'}, {w_orth_h_nmf_mu, w_orth_w_nmf_mu, nmf_hals_so}, {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_hals_so});



