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

% set ranks for each layer
%rank_layers = [100 50];
rank_layers = [150 10];

% set options
options.max_epoch = 100;
options.verbose = 1;


%% Deep-Semi-NMF
[w_deep_semi_nmf, infos_deep_semi_nmf] = deep_semi_nmf(V, rank_layers, options);
display_graph('iter','cost', {'Deep-SemiNMF'}, {w_deep_semi_nmf}, {infos_deep_semi_nmf});



%% Deep-Multi-Semi-NMF
[w_multi_deep_semi_nmf, infos_multi_deep_semi_nmf] = deep_multiview_semi_nmf(X, rank_layers, options);
display_graph('iter','cost', {'Deep-Multi-SemiNMF'}, {w_multi_deep_semi_nmf}, {infos_multi_deep_semi_nmf});






