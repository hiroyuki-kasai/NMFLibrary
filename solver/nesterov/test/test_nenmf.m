function test_nenmf(varargin)
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


    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 500;
        n = 100;
        V = rand(m,n);
        rank = 20;
        options = [];
        options.verbose = 2;
        options.max_epoch = 100;

        health_check_mode = false;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end

    % initialize factor matrices
    options.init_alg = 'kmeans';  
    [options.x_init, ~] = generate_init_factors(V, rank, []);      


    %% Calculate f_opt
    if ~health_check_mode
        fprintf('Calculating f_opt by HALS ...\n');
        options.alg = 'hals';
        options.max_epoch = 500;
        [w_sol, ~] = als_nmf(V, rank, options);
        f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(m, n));
        fprintf('Done.. f_opt: %.16e\n', f_opt);    
        options.f_opt = f_opt;
    end
    
    
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);

    % Hierarchical ALS
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = als_nmf(V, rank, options);   
    
    
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
    
    
    if ~health_check_mode      
        %% plot
        display_graph('epoch','optimality_gap', {'MU', 'HALS', 'NeNMF (Plain)', 'NeNMF (L1R)', 'NeNMF (L2R)', 'NeNMF (MR)'}, ...
                    {w_nmf_mu, w_nmf_hals, w_nenmf_p, w_nenmf_l1, w_nenmf_l2, w_nenmf_mr}, ...
                    {infos_nmf_mu, infos_nmf_hals, infos_nenmf_p, infos_nenmf_l1, infos_nenmf_l2, infos_nenmf_mr});
                
        display_graph('time','optimality_gap', {'MU', 'HALS', 'NeNMF (Plain)', 'NeNMF (L1R)', 'NeNMF (L2R)', 'NeNMF (MR)'}, ...
                    {w_nmf_mu, w_nmf_hals, w_nenmf_p, w_nenmf_l1, w_nenmf_l2, w_nenmf_mr}, ...
                    {infos_nmf_mu, infos_nmf_hals, infos_nenmf_p, infos_nenmf_l1, infos_nenmf_l2, infos_nenmf_mr});
    end
            
end