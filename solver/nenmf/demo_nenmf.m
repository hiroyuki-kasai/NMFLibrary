function demo_nenmf()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates NeNMF with multiplicative updates (MU) algorithm and 
% hierarchical alternative least squares (Hierarchical ALS) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on May 21, 2019

    %rng('default')

    clc;
    clear;
    close all;

    if 0
        %% generate synthetic data of (mxn) matrix     
        m = 500;
        n = 100;
        V = rand(m,n);
        rank = 5;
    else
        V = importdata('../../data/CBCL_face.mat');
        [m, n] = size(V);
        rank = 10;
    end
    
    
    %% Initialize of rank to be factorized
    

    
    %% Calculate f_opt
    fprintf('Calculating f_opt by HALS ...\n');
    options.alg = 'hals';
    options.max_epoch = 500;
    [w_sol, ~] = nmf_als(V, rank, options);
    f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(m, n));
    fprintf('Done.. f_opt: %.16e\n', f_opt);    



    %% perform factroization
%     W0 = rand(m, rank);
%     H0 = rand(rank, n);
%     x_init.W = W0;
%     x_init.H = H0;   
    x_init = [];
    options.init_alg = 'kmeans';
    options.x_init = x_init;
    options.verbose = 2;
    options.f_opt = f_opt;
    
    
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);

    % Hierarchical ALS
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);   
    
    
    % NeNMF
    options.lambda = 1;
    
    options.type = 'plain';
    [w_nenmf_p, infos_nenmf_p] = nenmf(V, rank, options);
    
    options.type = 'l1r';
    [w_nenmf_l1, infos_nenmf_l1] = nenmf(V, rank, options);
    
    options.type = 'l2r';
    [w_nenmf_l2, infos_nenmf_l2] = nenmf(V, rank, options);
    
    options.type = 'mr';
    options.sim_mat = constructW(V');
    [w_nenmf_mr, infos_nenmf_mr] = nenmf(V, rank, options);    
    
    
    %% plot
    display_graph('epoch','optimality_gap', {'MU', 'HALS', 'NeNMF (Plain)', 'NeNMF (L1R)', 'NeNMF (L2R)', 'NeNMF (MR)'}, ...
                {w_nmf_mu, w_nmf_hals, w_nenmf_p, w_nenmf_l1, w_nenmf_l2, w_nenmf_mr}, ...
                {infos_nmf_mu, infos_nmf_hals, infos_nenmf_p, infos_nenmf_l1, infos_nenmf_l2, infos_nenmf_mr});
            
    display_graph('time','optimality_gap', {'MU', 'HALS', 'NeNMF (Plain)', 'NeNMF (L1R)', 'NeNMF (L2R)', 'NeNMF (MR)'}, ...
                {w_nmf_mu, w_nmf_hals, w_nenmf_p, w_nenmf_l1, w_nenmf_l2, w_nenmf_mr}, ...
                {infos_nmf_mu, infos_nmf_hals, infos_nenmf_p, infos_nenmf_l1, infos_nenmf_l2, infos_nenmf_mr});
            

    
end


