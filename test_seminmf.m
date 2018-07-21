function test_seminmf()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates semi-nmf algorithm and 
% hierarchical alternative least squares (Hierarchical ALS) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 21, 2017

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

    %% perform factroization
    % Semi-NMF
    [w_seminmf_mu, infos_seminmf_mu] = semi_nmf(V, rank, options);
    
    % Hierarchical ALS
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);       
    
    
    %% plot
    display_graph('epoch','cost', {'Semi-NMF', 'HALS'}, {w_seminmf_mu, w_nmf_hals}, {infos_seminmf_mu, infos_nmf_hals});
    display_graph('time','cost', {'Semi-NMF', 'HALS'}, {w_seminmf_mu, w_nmf_hals}, {infos_seminmf_mu, infos_nmf_hals});
    
end


