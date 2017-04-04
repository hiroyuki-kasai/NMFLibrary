function demo_face()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates multiplicative updates (MU) algorithm and 
% hierarchical alternative least squares (Hierarchical ALS) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Apr. 05, 2017

    clc;
    clear;
    close all;

    %% load CBCL face datasets
    V = importdata('./data/CBCL_face.mat');

    
    %% Initialize of rank to be factorized
    rank = 5;
    
    
    %% Set options
    options.verbose = 1;


    %% perform factroization
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
    % Hierarchical ALS
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);       
    
    
    %% plot
    display_graph('epoch','cost', {'MU', 'HALS'}, {w_nmf_mu, w_nmf_hals}, {infos_nmf_mu, infos_nmf_hals});
    display_graph('time','cost', {'MU', 'HALS'}, {w_nmf_mu, w_nmf_hals}, {infos_nmf_mu, infos_nmf_hals});
    
    
    %% display basis elements obtained with different algorithms
    plot_dictionnary(w_nmf_mu.W, [], [7 7]); 
    plot_dictionnary(w_nmf_hals.W, [], [7 7]); 
end


