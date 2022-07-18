function test_rank2nmf(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates rank2nmf algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 14, 2022

    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        options = [];
        options.verbose = 2;
        options.max_epoch = 100; 
        health_check_mode = false;
    else
        options = varargin{3};
        health_check_mode = true;
    end 

    m = 500;
    n = 100;    
    V = rand(m,2) * rand(2,n) + 0.05 * rand(m,n); 
    rank = 2;    

    %%% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        


    % new solver
    options.alg = 'acc_hals';    
    [~, info_rank2] = rank2nmf(V, options);    
     
    
end