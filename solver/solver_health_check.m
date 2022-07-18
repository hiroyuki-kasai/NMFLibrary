function solver_health_check()
%
% health-check script for algorithms in NMFLibrary
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 08, 2022

    clc;
    clear;
    close all;

    % generate synthetic data of (mxn) matrix       
    m = 500;
    n = 100;
    V = rand(m, n);
    rank = 10;

    % set options
    health_options = [];
    health_options.verbose = 1; 
    health_options.max_epoch = 100;


    % execute test scripts for algorithms
    test_separable(V, rank, health_options);
    test_prob_nmf(V, rank, health_options);
    test_sparse_nmf(V, rank, health_options);    
    test_semi_nmf(V, rank, health_options);    
    test_online_nmf(V, rank, health_options);
    test_deep_nmf(V, rank, health_options);
    test_conv_nmf(V, rank, health_options);
    test_nmtf(V, rank, health_options);   
    test_minvolnmf(V, rank, health_options);
    test_recursive_nmu(V, rank, health_options);
    test_rank2nmf(V, rank, health_options);
    test_wlra(V, rank, health_options); 
    test_projective_nmf(V, rank, health_options); 
    test_convex_nmf(V, rank, health_options);     
    test_orth_nmf(V, rank, health_options);   
    test_robust_nmf(V, rank, health_options);
    test_prob_nmf(V, rank, health_options);
    test_divergence_nmf(V, rank, health_options);
    test_nenmf(V, rank, health_options);    
    test_orth_nmf(V, rank, health_options);      
    test_mu(V, rank, health_options);    
    test_pgd(V, rank, health_options);
    test_als(V, rank, health_options);    
    test_anls(V, rank, health_options);
    test_admm(V, rank, health_options);  

    fprintf('done.\n'); 

end