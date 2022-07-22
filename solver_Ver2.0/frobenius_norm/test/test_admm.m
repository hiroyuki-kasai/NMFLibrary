function test_admm(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 21, 2020


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
    inner_max_epoch = 500;      
    rescale = true;

    W = x_init.W;
    H = x_init.H;     
    if rescale
        [W, H] = normalize_WH(V, W, H, rank, 'type1');       
    end

    x_init.W = W;
    x_init.H = H;      

    
    %% ADMM
    % standard
    options.inner_max_epoch = 1;
    [w_nmf_admm, infos_nmf_admm] = admm_nmf(V, rank, options);  
    
    % acc only H
    options.momentum_h = 3;
    options.momentum_w = 0;    
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    [w_nmf_admm_momentum_h, infos_nmf_admm_momentum_h] = admm_nmf(V, rank, options);   
    
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    [w_nmf_admm_momentum_hw, infos_nmf_admm_momentum_hw] = admm_nmf(V, rank, options);  
%     
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.x_init = x_init;
    options.inner_max_epoch = inner_max_epoch;
    options.scaling = false;    
    [w_nmf_admm_momentum_hw_inner, infos_nmf_admm_momentum_hw_inner] = admm_nmf(V, rank, options);     
    
 
    if ~health_check_mode      
        %% plot
        display_graph('iter','cost', {'ADMM-Standard', 'ADMM-MOMENTUM (H)', 'ADMM-MOMENTUM (W,H)', 'ADMM-MOMENTUM-nonscale (W,H)'}, ...
                                {w_nmf_admm, w_nmf_admm_momentum_h, w_nmf_admm_momentum_hw, w_nmf_admm_momentum_hw_inner}, ...
                                {infos_nmf_admm, infos_nmf_admm_momentum_h, infos_nmf_admm_momentum_hw, infos_nmf_admm_momentum_hw_inner});
    end

    
end