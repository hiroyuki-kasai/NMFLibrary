function test_pgd(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 21, 2020
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
    [x_init, ~] = generate_init_factors(V, rank, []);      
    inner_max_epoch = 10;  

    
    %% PGD
    algorithm = 'pgd';    
    % algorithm = 'fast_pgd';
    %algorithm = 'adp_step_pgd';
    
    %options.alg = 'direct_pgd';      
    %[w_nmf_pgd, infos_nmf_pgd] = pgd_nmf(V, rank, options);     
    
    % standard
    options.momentum_h = 0;
    options.momentum_w = 0;    
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;   
    [w_nmf_pgd, infos_nmf_pgd] = pgd_nmf(V, rank, options);  
    
    % acc only H
    options.momentum_h = 3;
    options.momentum_w = 0;    
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;    
    [w_nmf_pgd_momentum_h, infos_nmf_pgd_momentum_h] = pgd_nmf(V, rank, options);   
    
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;    
   [w_nmf_pgd_momentum_hw, infos_nmf_pgd_momentum_hw] = pgd_nmf(V, rank, options);  
    
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    options.scaling = false; 
    options.alg = algorithm;
    [w_nmf_pgd_momentum_hw_inner, infos_nmf_pgd_momentum_hw_inner] = pgd_nmf(V, rank, options); 

   
    if ~health_check_mode
        %% plot
        display_graph('iter','cost', {'PGD-Standard', 'PGD-MOMENTUM (H)', 'PGD-MOMENTUM (W,H)', 'PGD-MOMENTUM-nonscale (W,H)'}, ...
                                {w_nmf_pgd, w_nmf_pgd_momentum_h, w_nmf_pgd_momentum_hw, w_nmf_pgd_momentum_hw_inner}, ...
                                {infos_nmf_pgd, infos_nmf_pgd_momentum_h, infos_nmf_pgd_momentum_hw, infos_nmf_pgd_momentum_hw_inner});
    end

    
end