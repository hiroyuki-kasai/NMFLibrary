function test_deep_nmf(varargin)
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Apr. 05, 2017

    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 300;
        n = 500;
        V = rand(m,n);
        options = [];
        options.verbose = 1;
        options.max_epoch = 100; 
        health_check_mode = false;
    else
        V = varargin{1};
        %rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end

    
    %% Initialize of rank to be factorized
    rank_layers = [49 25 16];
    
    %% Deep-Semi-NMF
    [w_deep_semi_nmf, infos_deep_semi_nmf] = deep_semi_nmf(V, rank_layers, options);
    
    %% Deep-ns-NMF
    options.theta = 0.5;
    %options.update_alg = 'mu';
    options.update_alg = 'apg';
    options.apg_maxiter = 10;
    [w_deep_ns_nmf, infos_deep_ns_nmf] = deep_ns_nmf(V, rank_layers, options); 

    [w_deep_bi_nmf, infos_deep_bi_nmf] = deep_bidirectional_nmf(V, rank_layers, options); 
    

    
    
    %% Plotting
    if ~health_check_mode        
        display_graph('iter','cost', {'Deep-SemiNMF', 'Deep-nsNMF', 'Deep-Bidir-SemiNMF'}, {w_deep_semi_nmf, w_deep_ns_nmf, w_deep_bi_nmf}, {infos_deep_semi_nmf, infos_deep_ns_nmf, infos_deep_bi_nmf});
    end

        
end


