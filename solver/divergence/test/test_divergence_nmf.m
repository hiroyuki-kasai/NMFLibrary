function test_divergence_nmf(varargin)
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 24, 2017
%
% Change log: 
%
%   Jun. 27, 2022 (Hiroyuki Kasai): Modified descriptions.
%
%


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
    options.init_alg = 'NNDSVD';  
    [options.x_init, ~] = generate_init_factors(V, rank, []);   

    
    %% perform factroization
    % 'euc';
    [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);
    
    % 'kl-div';
    options.metric_type = 'kl-div';
    [w_nmf_mu_kl, infos_nmf_mu_kl] = div_mu_nmf(V, rank, options);
    
    options.metric_type = 'kl-div';
    [w_kl_bmd_nmf, infos_kl_bmd_nmf] = kl_bmd_nmf(V, rank, options);
    
    options.metric_type = 'kl-div';
    [w_nmf_kl_fpa, infos_nmf_kl_fpa] = kl_fpa_nmf(V, rank, options);
    
    options.metric_type = 'beta-div';
    options.d_beta = 1;
    [w_nmf_kl_admm, infos_nmf_kl_admm] = div_admm_nmf(V, rank, options);
    
    % 'alpha-div';
    options.metric_type = 'alpha-div'; % Neyman's chi-square distance
    [w_nmf_mu_alpha_m1, infos_nmf_mu_alpha_m1] = div_mu_nmf(V, rank, options);
    
    options.d_alpha = 0.5; % Hellinger's distance
    [w_nmf_mu_alpha05, infos_nmf_mu_alpha05] = div_mu_nmf(V, rank, options);
    
    options.d_alpha = 2; % Pearson's distance
    [w_nmf_mu_alpha2, infos_nmf_mu_alpha2] = div_mu_nmf(V, rank, options);
    
    % 'beta-div';
    options.metric_type = 'beta-div';
    [w_nmf_mu_beta0, infos_nmf_mu_beta0] = div_mu_nmf(V, rank, options);
    
    options.d_beta = 1;
    [w_nmf_mu_beta1, infos_nmf_mu_beta1] = div_mu_nmf(V, rank, options);
    
    options.d_beta = 2;
    [w_nmf_mu_beta2, infos_nmf_mu_beta2] = div_mu_nmf(V, rank, options);
    
    options.metric_type = 'beta-div';
    options.d_beta = 0;
    options.rho = 500;
    [w_nmf_is_admm, infos_nmf_is_admm] = div_admm_nmf(V, rank, options);
        
    
    if ~health_check_mode      
        %% plot
            display_graph('epoch','cost',{'Fro-MU (euc)','div-MU (kl-div)','KL-BMD','KL-FPA', ...
                'div-ADMM (beta=1: KL)','div-MU (alpha=-1)','div-MU (alpha=0.5)','div-MU (alpha=2)',...
                'div-MU (beta=0: IS)','div-MU (beta=1: kl-div)','div-MU (beta=2: Frobenius Norm)',...
                'div-ADMM (beta=0: IS)'}, ...
                [], ...
                {infos_nmf_mu,infos_nmf_mu_kl,infos_kl_bmd_nmf,infos_nmf_kl_fpa,infos_nmf_kl_admm,...
                infos_nmf_mu_alpha_m1,infos_nmf_mu_alpha05,infos_nmf_mu_alpha2,...
                infos_nmf_mu_beta0,infos_nmf_mu_beta1,infos_nmf_mu_beta2,infos_nmf_is_admm});


    end

end

