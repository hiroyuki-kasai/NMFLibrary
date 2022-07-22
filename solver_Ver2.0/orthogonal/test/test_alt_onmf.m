% function test_alt_onmf()
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 08, 2022

    clc;
    clear;
    close all;

    %rng('default')

    %% generate synthetic data of (mxn) matrix       
    m = 500;
    n = 500;
    V = rand(m,n);
    rank = 50;
    
    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        

    %% perform factroization
    options.verbose = 2;
    options.delta = -Inf;
    [x, info_new] = alternating_onmf(V, rank, options);


