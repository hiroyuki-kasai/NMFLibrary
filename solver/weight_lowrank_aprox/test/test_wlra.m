function test_wlra(varargin)
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 08, 2022

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
    
    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        

    %% perform factroization
    P = round(rand(size(V)));  
    
    [x, info_wla] = wlra(V, rank, P, options);

end
