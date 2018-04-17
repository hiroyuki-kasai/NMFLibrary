function comp_nmf_base_algorithms()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on OCt. 27, 2017

    clc;
    clear;
    close all;

    %% generate synthetic data of (mxn) matrix       
    m = 1000;
    n = 100;
    V = rand(m,n);
    
    
    %% Initialize rank to be factorized
    rank = 50;
    options.verbose = 2;


    
    %% MU variants
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);  

    % MU mod
    options.alg = 'mu_mod';
    [w_nmf_mu_mod, infos_nmf_mu_mod] = nmf_mu(V, rank, options); 
    
    % Accelerated MU
    options.alg = 'mu_acc';
    [w_nmf_mu_acc, infos_nmf_mu_acc] = nmf_mu(V, rank, options);     
    
    
    %% ALS variants
    % ALS
    options.alg = 'als';
    [w_nmf_als, infos_nmf_als] = nmf_als(V, rank, options);       
    
    % Hierarchical ALS    
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);
    
    % Accelerated Hierarchical ALS    
    options.alg = 'acc_hals';
    [w_nmf_acc_hals, infos_nmf_acc_hals] = nmf_als(V, rank, options); 
    
    
    %% PGD variants
    % PGD
    options.alg = 'pg';
    [w_nmf_pgd, infos_nmf_pgd] = nmf_pgd(V, rank, options);    
    
    % PGD direct    
    options.alg = 'direct_pgd';
    [w_nmf_pgd_dir, infos_nmf_pgd_dir] = nmf_pgd(V, rank, options);     
    
    
    
    %% plot
    display_graph('iter','cost', {'MU', 'MU-Mod', 'MU-ACC', 'ALS', 'HALS', 'HALS-ACC', 'PGD', 'PGD-DIR'}, ...
                                {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc, w_nmf_als, w_nmf_hals, w_nmf_acc_hals, w_nmf_pgd, w_nmf_pgd_dir}, ...
                                {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc, ...
                                infos_nmf_als, infos_nmf_hals, infos_nmf_acc_hals, ...
                                infos_nmf_pgd, infos_nmf_pgd_dir});
    display_graph('time','cost', {'MU', 'MU-Mod', 'MU-ACC', 'ALS', 'HALS', 'HALS-ACC', 'PGD', 'PGD-DIR'}, ...
                                {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc, w_nmf_als, w_nmf_hals, w_nmf_acc_hals, w_nmf_pgd, w_nmf_pgd_dir}, ...
                                {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc, ...
                                infos_nmf_als, infos_nmf_hals, infos_nmf_acc_hals, ...
                                infos_nmf_pgd, infos_nmf_pgd_dir});
    
end


