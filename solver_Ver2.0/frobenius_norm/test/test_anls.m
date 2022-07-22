function test_anls(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 23, 2022

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
    rescale = true;
    
    
    %% Scale initialization so that argmin_a ||a * WH - X||_F = 1   
    W = x_init.W;
    H = x_init.H;     
    if rescale
        [W, H] = normalize_WH(V, W, H, rank, 'type1');       
    end

    x_init.W = W;
    x_init.H = H;    
    
    
    %% ANLS
    %algorithm = 'anls_bpp';
    %algorithm = 'anls_asgroup';   
    algorithm = 'anls_asgivens';       
    % standard
    options.momentum_h = 0;
    options.momentum_w = 0;     
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;   
 
    [w_nmf_anls, infos_nmf_anls] = anls_nmf(V, rank, options);  
    
    % acc only H
    options.momentum_h = 3;
    options.momentum_w = 0;    
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;    
    [w_nmf_anls_momemtum_h, infos_nmf_anls_momemtum_h] = anls_nmf(V, rank, options);   
    
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.inner_max_epoch = inner_max_epoch;
    options.alg = algorithm;    
    [w_nmf_anls_momemtum_hw, infos_nmf_anls_momemtum_hw] = anls_nmf(V, rank, options);  
    
    options.momentum_h = 3;
    options.momentum_w = 2; 
    options.inner_max_epoch = inner_max_epoch;
    options.scaling = false; 
    options.alg = algorithm;
    [w_nmf_anls_momemtum_hw_inner, infos_nmf_anls_momemtum_hw_inner] = anls_nmf(V, rank, options);     
    

    if ~health_check_mode

        %% plot
        display_graph('iter','cost', {'ANLS-Standard', 'ANLS-MOMENTUM (H)', 'ANLS-MOMENTUM (W,H)', 'ANLS-MOMENTUM-nonscale (W,H)'}, ...
                                {w_nmf_anls, w_nmf_anls_momemtum_h, w_nmf_anls_momemtum_hw, w_nmf_anls_momemtum_hw_inner}, ...
                                {infos_nmf_anls, infos_nmf_anls_momemtum_h, infos_nmf_anls_momemtum_hw, infos_nmf_anls_momemtum_hw_inner});
    end   
    
    
end