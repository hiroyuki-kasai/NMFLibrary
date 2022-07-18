function demo_face()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates Frobenius-norm based multiplicative updates (MU) algorithm and 
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
    % Fro-MU
    options.alg = 'mu';
    [w_mu, infos_mu] = fro_mu_nmf(V, rank, options);
    % Hierarchical ALS
    options.alg = 'hals';
    [w_hals, infos_hals] = als_nmf(V, rank, options);       
    
    
    %% plot
    display_graph('epoch','cost', {'MU', 'HALS'}, {w_mu, w_hals}, {infos_mu, infos_hals});
    display_graph('time','cost', {'MU', 'HALS'}, {w_mu, w_hals}, {infos_mu, infos_hals});
    
    
    %% display basis elements obtained with different algorithms
    plot_dictionnary(w_mu.W, [], [7 7]); 
    plot_dictionnary(w_hals.W, [], [7 7]); 
end


