function test_orth_nmf(varargin)
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 23, 2017
%
% Change log: 
%
%   May  16, 2019 (Hiroyuki Kasai): Fixed algorithm.
%
%   Jun. 21, 2021 (Hiroyuki Kasai): Changed the initialization method.

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
    [options.x_init, ~] = generate_init_factors(V, rank, []);      
    
    
    %% perform factroization
    % NMF-MU
    %[w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);
    
    % Orth-MU for H
    options.orth_h    = 1;
    options.norm_h    = 2;
    options.orth_w    = 0;
    options.norm_w    = 0;    
    [w_orth_h_nmf_mu, infos_orth_h_nmf_mu] = orth_mu_nmf(V, rank, options);
    
    % Orth-MU for W
    options.orth_h    = 0;
    options.norm_h    = 0;
    options.orth_w    = 1;
    options.norm_w    = 2; 
    [w_orth_w_nmf_mu, infos_orth_w_nmf_mu] = orth_mu_nmf(V, rank, options);
    
    
    % DTPP
    options.norm_h    = 0;
    options.norm_w    = 0;
    
    % DTPP for H
    options.orth_h    = 1;
    options.orth_w    = 0;
    [w_nmf_orth_dtpp_h, infos_nmf_orth_dtpp_h] = dtpp_nmf(V, rank, options);
    
    % DTPP for W
    options.orth_h    = 0;
    options.orth_w    = 1;
    [w_nmf_orth_dtpp_w, infos_nmf_orth_dtpp_w] = dtpp_nmf(V, rank, options);
    
    % DTPP for W & H
    options.orth_h    = 1;
    options.orth_w    = 1;
    [w_nmf_orth_dtpp_wh, infos_nmf_orth_dtpp_wh] = dtpp_nmf(V, rank, options);
    
    
    % NMF-HALS-SO
    options.wo = 1;
    [w_nmf_hals_so, infos_nmf_hals_so] = hals_so_nmf(V, rank, options);

    % ALT-ONMF
    [w_nmf_alt_onmf, infos_alt_onmf] = alternating_onmf(V, rank, options);    
    
    
    if ~health_check_mode       
        %% plot
        display_graph('epoch','cost', {'Orth-MU-H', 'Orth-MU-W', 'DTPP-H', 'DTPP-W', 'DTPP-WH', 'NMF-HALS-SO', 'ALT-ONMF'}, ...
            {w_orth_h_nmf_mu, w_orth_w_nmf_mu, w_nmf_orth_dtpp_h, w_nmf_orth_dtpp_w, w_nmf_orth_dtpp_wh, w_nmf_hals_so,w_nmf_alt_onmf}, ...
            {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_orth_dtpp_h, infos_nmf_orth_dtpp_w, infos_nmf_orth_dtpp_wh, infos_nmf_hals_so,infos_alt_onmf});
        display_graph('epoch','orth', {'Orth-MU-H', 'Orth-MU-W', 'DTPP-H', 'DTPP-W', 'DTPP-WH', 'NMF-HALS-SO', 'ALT-ONMF'}, ...
            {w_orth_h_nmf_mu, w_orth_w_nmf_mu, w_nmf_orth_dtpp_h, w_nmf_orth_dtpp_w, w_nmf_orth_dtpp_wh, w_nmf_hals_so,w_nmf_alt_onmf}, ...
            {infos_orth_h_nmf_mu, infos_orth_w_nmf_mu, infos_nmf_orth_dtpp_h, infos_nmf_orth_dtpp_w, infos_nmf_orth_dtpp_wh, infos_nmf_hals_so,infos_alt_onmf});
    end

end

