function test_minvolnmf(varargin)
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
    
    %% perform factroization
    [w, info] = minvol_nmf(V, rank, options);
    
 
end
