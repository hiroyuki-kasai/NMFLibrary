function test_recursive_nmu(varargin)
%
% demonstration file for NMFLibrary.
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 06, 2022

  
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
    [~, info_rnmnew] = recursive_nmu(V, rank, options);

    if ~health_check_mode       
        display_graph('epoch','cost', {'Recursive-NMU'}, [], {info_rnmnew});
    end

end


