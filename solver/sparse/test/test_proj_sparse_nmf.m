% function demo_pro_semi_nmf()
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 15, 2022

    clc;
    clear;
    close all;
    
    %rng('default')

    %% generate synthetic data of (mxn) matrix       
    m = 100;
    n = 100;
    %V = randn(m,n);
    V = randn(m,n);
    maxiter = 100;    
    options.verbose = 2;
    options.max_epoch = maxiter;
    rank = 5;

    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        

    %% perform factroization
    [sol_1, info_1] = proj_sparse_nmf(V, rank, options);

    options.sW = 0.8;
    [sol_2, info_2] = proj_sparse_nmf(V, rank, options);   

    options.sH = 0.8;     
    options.FPGM = true;    
    [sol_3, info_3] = proj_sparse_nmf(V, rank, options);        
    
    
    %% plot
    display_graph('epoch','cost', {'Standard', 'Sparse W', 'Sparse W+H'}, {info_1, info_2, info_3}, {info_1, info_2, info_3});

    display_sparsity_graph({'Standard', 'Sparse W', 'Sparse H'},{sol_1, sol_2, sol_3});    
    


