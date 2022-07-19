function demo()
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

    %% generate synthetic data of (mxn) matrix       
    m = 500;
    n = 100;
    V = rand(m,n);
    
    
    %% Initialize of rank to be factorized
    rank = 5;


    %% perform factroization
    options.verbose = 1;
    % MU
    options.alg = 'mu';
    [w_mu, infos_mu] = fro_mu_nmf(V, rank, options);
    % Hierarchical ALS
    options.alg = 'hals';
    [w_hals, infos_hals] = als_nmf(V, rank, options);    
    % Accelerated Hierarchical ALS
    options.alg = 'acc_hals';
    [w_acchals, infos_acchals] = als_nmf(V, rank, options);      
    
    
    %% plot
    display_graph('epoch','cost', {'Fro-MU', 'HALS', 'Acc-HALS'}, {w_mu, w_hals, w_acchals}, {infos_mu, infos_hals, infos_acchals});
    display_graph('time','cost', {'Fro-MU', 'HALS', 'Acc-HALS'}, {w_mu, w_hals, w_acchals}, {infos_mu, infos_hals, infos_acchals});
    
end