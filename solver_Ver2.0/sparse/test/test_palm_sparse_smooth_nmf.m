function test_palm()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates multiplicative updates (MU) algorithm and 
% hierarchical alternative least squares (Hierarchical ALS) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Apr. 05, 2017

    clc;
    clear;
    close all;
    
    rng('default')    

    %% generate synthetic data of (mxn) matrix       
    m = 500;
    n = 100;
    V = rand(m, n);
    options.verbose = 2;
    options.max_epoch = 100;

    rank = 5;

    
    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);
    
    options.lambda = 0;
    options.eta = 0;    
    [w_std, info_std] = palm_sparse_smooth_nmf(V, rank, options);

    options.lambda = 0;
    options.eta = 0.5;    
    [w_smooth, info_smooth] = palm_sparse_smooth_nmf(V, rank, options);

    options.lambda = 2.5;
    options.eta = 0;    
    [w_sparse, info_sparse] = palm_sparse_smooth_nmf(V, rank, options);

    options.lambda = 2.5;
    options.eta = 0.5;    
    [w_sparse_smooth, info_sparse_smooth] = palm_sparse_smooth_nmf(V, rank, options);    

    display_graph('epoch','cost', {'Standard', 'Smooth', 'Sparse', 'Sparse & Smooth'}, ...
        [], {info_std, info_smooth, info_sparse, info_sparse_smooth});

    display_sparsity_graph({'Standard', 'Smooth', 'Sparse', 'Sparse & Smooth'},...
        {w_std, w_smooth,w_sparse,w_sparse_smooth});    
    
    
    
end


