function test_mu(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on OCt. 27, 2017

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

    % initialize factor matrices
    [x_init, ~] = generate_init_factors(V, rank, []); 
    options.x_init = x_init;    
    
    %% MU variants
    % MU
    options.alg = 'mu';
    [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);  

    % MU mod
    options.alg = 'mu_mod';
    [w_nmf_mu_mod, infos_nmf_mu_mod] = fro_mu_nmf(V, rank, options); 
    
    % Accelerated MU
    options.alg = 'mu_acc';
    [w_nmf_mu_acc, infos_nmf_mu_acc] = fro_mu_nmf(V, rank, options);     
    
    
    if ~health_check_mode
        %% plot
        display_graph('iter','cost', {'MU', 'MU-Mod', 'MU-ACC'}, {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc},  {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc});
        display_graph('time','cost', {'MU', 'MU-Mod', 'MU-ACC'}, {w_nmf_mu, w_nmf_mu_mod, w_nmf_mu_acc},  {infos_nmf_mu, infos_nmf_mu_mod, infos_nmf_mu_acc});
    end
    
end