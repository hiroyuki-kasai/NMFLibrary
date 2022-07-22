function test_separable(varargin)
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 07, 2022

    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 1500;
        n = 100;
        V = rand(m,n);
        rank = 10;
        options = [];
        options.verbose = 1;
        health_check_mode = false;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end

    %options.normalize = 1;
    options.precision = 1e-6;    
    %options.precision = 1;      
    [x_spa, info_spa] = spa(V, rank, options);  

    options.inner_max_epoch = 10;
    %options.inner_nnls_alg = 'hals';
    [x_snpa, info_snpa] = snpa(V, rank, options);        

end