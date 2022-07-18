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
    V = randn(m,n);
    options.verbose = 2;
    options.max_epoch = 10;
    rank = 5;

    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        

    %% perform factroization
    options.sW = 0;
    [sol_000, info_000] = proj_sparse_nmf(V, rank, options);

    options.sW = 0.25;
    [sol_025, info_025] = proj_sparse_nmf(V, rank, options);    

    options.sW = 0.5;
    [sol_050, info_050] = proj_sparse_nmf(V, rank, options);  

    options.sW = 0.75;
    [sol_075, info_075] = proj_sparse_nmf(V, rank, options);  

    options.sW = 1;
    [sol_100, info_100] = proj_sparse_nmf(V, rank, options);     
    
    %% plot
    display_graph('epoch','cost', {'s=0.00', 's=0.25', 's=0.50', 's=0.75', 's=1.00'}, ...
        {info_000, info_025, info_050, info_075, info_100}, ...
        {info_000, info_025, info_050, info_075, info_100});

    display_sparsity_graph({'s=0.00', 's=0.25', 's=0.50', 's=0.75', 's=1.00'},...
        {sol_000, sol_025, sol_050, sol_075, sol_100});    
    


