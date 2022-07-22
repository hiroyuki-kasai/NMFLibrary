function test_frobenius_norm_synth()
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
    
    
    %% initialize rank to be factorized
    rank = 50;
    
    options.verbose = 1;
    options.max_epoch = 20;
    
    % initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);       

    
    %% MU variants
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);  

    % MU mod
    options.alg = 'mu_mod';
    [w_nmf_mu_mod, infos_nmf_mu_mod] = fro_mu_nmf(V, rank, options); 
    
    % Accelerated MU
    options.alg = 'mu_acc';
    [w_nmf_mu_acc, infos_nmf_mu_acc] = fro_mu_nmf(V, rank, options);     
    
    
    %% ALS variants
    % ALS
    options.alg = 'als';
    [w_nmf_als, infos_nmf_als] = als_nmf(V, rank, options);       
    
    % Hierarchical ALS    
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = als_nmf(V, rank, options);
    
    % Accelerated Hierarchical ALS    
    options.alg = 'acc_hals';
    [w_nmf_acc_hals, infos_nmf_acc_hals] = als_nmf(V, rank, options); 
    
    
    %% PGD variants
    % PGD
    options.alg = 'pgd';
    [w_nmf_pgd, infos_nmf_pgd] = pgd_nmf(V, rank, options);  
    
    % Fast PGD
    options.alg = 'fast_pgd'; 
    options.inner_max_epoch = 1;
    [w_nmf_fpgd, infos_nmf_fpgd] = pgd_nmf(V, rank, options);        
    
    % Step-adaptive PGD
    options.alg = 'adp_step_pgd';
    [w_nmf_as_pgd, infos_nmf_as_pgd] = pgd_nmf(V, rank, options);  
    
    % PGD direct    
    options.alg = 'direct_pgd';
    [w_nmf_pgd_dir, infos_nmf_pgd_dir] = pgd_nmf(V, rank, options);     
    
    
    
    %% plot
    display_graph('iter','cost', {'MU', 'MU-Mod', 'MU-ACC', 'ALS', 'HALS', 'HALS-ACC', 'PGD', 'FPGD', 'ADSTEP-PGD', 'DIR-PGD'}, ...
                                {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc, w_nmf_als, w_nmf_hals, w_nmf_acc_hals, w_nmf_pgd, w_nmf_fpgd, w_nmf_as_pgd, w_nmf_pgd_dir}, ...
                                {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc, ...
                                infos_nmf_als, infos_nmf_hals, infos_nmf_acc_hals, ...
                                infos_nmf_pgd, infos_nmf_fpgd, infos_nmf_as_pgd, infos_nmf_pgd_dir});
    display_graph('time','cost', {'MU', 'MU-Mod', 'MU-ACC', 'ALS', 'HALS', 'HALS-ACC', 'PGD', 'FPGD', 'ADSTEP-PGD', 'DIR-PGD'}, ...
                                {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc, w_nmf_als, w_nmf_hals, w_nmf_acc_hals, w_nmf_pgd, w_nmf_fpgd, w_nmf_as_pgd, w_nmf_pgd_dir}, ...
                                {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc, ...
                                infos_nmf_als, infos_nmf_hals, infos_nmf_acc_hals, ...
                                infos_nmf_pgd, infos_nmf_fpgd, infos_nmf_as_pgd, infos_nmf_pgd_dir});
    
end