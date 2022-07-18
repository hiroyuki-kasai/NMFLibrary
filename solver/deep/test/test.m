% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 26, 2017

clc;
clear;
close all;

% load datasets
load ../../../data/yale_mtv.mat;
X{1,1} = NormalizeFea(X{1,1}, 0);
X{1,2} = NormalizeFea(X{1,2}, 0);
X{1,3} = NormalizeFea(X{1,3}, 0);
V = X{1,1};
gnd = gt;


% m = 500;
% n = 200;
% V = rand(m,n);

% set ranks for each layer
%rank_layers = [150 100 80 60 40 20];
rank_layers = [20];
%rank_layers = [150 120 90 50 30];

% set options
options.max_epoch = 100;
options.verbose = 2;

% %% NMF
% options.alg = 'mu';
% [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank_layers(end), options);
% % Hierarchical ALS
% options.alg = 'hals';
% [w_nmf_hals, infos_nmf_hals] = als_nmf(V, rank_layers(end), options);   
% % nsNMF
% options.theta = 0.8;
% options.update_alg = 'apg';
% [w_ns_nmf, infos_ns_nmf] =  ns_nmf(V, rank_layers(end), options); 
% 
% %% Deep-ns-NMF
% options.theta = 0.8;
% options.apg_maxiter = 100;
% %options.update_alg = 'mu';
% options.update_alg = 'apg';
% [w_deep_ns_nmf, infos_deep_ns_nmf] = deep_ns_nmf(V, rank_layers, options);
% % fprintf('\n\n');
% 
% %return;
% 
% %% Deep-Semi-NMF
% %options.updateH_alg = 'mu';
% options.updateH_alg = 'apg';
% [w_deep_nmf_apg, infos_deep_nmf_apg] = deep_semi_nmf(V, rank_layers, options);
% fprintf('\n\n');
% 
% 
% display_graph('iter','cost', {'NMF(MU)', 'NMF(HALS)','nsNMF','Deep-nsNMF','Deep-NMF (APG)'}, ...
%     {w_nmf_mu, w_nmf_hals, w_ns_nmf, w_deep_ns_nmf, w_deep_nmf_apg}, ...
%     {infos_nmf_mu, infos_nmf_hals, infos_ns_nmf, infos_deep_ns_nmf, infos_deep_nmf_apg});
% 


%% Deep-Multi-Semi-NMF
[w_multi_deep_semi_nmf, infos_multi_deep_semi_nmf] = deep_multiview_semi_nmf(X, rank_layers, options);
display_graph('iter','cost', {'Deep-Multi-SemiNMF'}, {w_multi_deep_semi_nmf}, {infos_multi_deep_semi_nmf});






