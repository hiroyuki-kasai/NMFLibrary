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
m = 500;
n = 100;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 5;


%% Initialize of W and H
options = [];
[x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
options.x_init = x_init;
options.verbose = 2;
options.max_epoch = 100;



%% perform factroization
% NMF-MU
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);

% Orth-MU
options.orth_h    = 1;
options.norm_h    = 1;
options.orth_w    = 0;
options.norm_w    = 0;    
[w_orth_nmf_mu, infos_orth_nmf_mu] = nmf_orth_mu(V, rank, options);


%% plot
display_graph('epoch','cost', {'NMF-MU', 'Orth-MU'}, {w_nmf_mu, w_orth_nmf_mu}, {infos_nmf_mu, infos_orth_nmf_mu});
display_graph('time','cost', {'NMF-MU','Orth-MU'}, {w_nmf_mu, w_orth_nmf_mu}, {infos_nmf_mu, infos_orth_nmf_mu});



