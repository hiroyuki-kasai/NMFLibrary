function test_semi_nmf(varargin)
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 7, 2022
%
% Change log: 
%


    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 500;
        n = 200;
        %V = rand(m,n);
        V = randn(m,n);
        options.verbose = 2;
        options.max_epoch = 100;
        rank = 20;
    
        health_check_mode = false;
    else
        %V = varargin{1};
        m = 500;
        n = 200;
        %V = rand(m,n);
        V = randn(m,n);

        rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end


    %% Initialize factor matrices
    [options.x_init, ~] = generate_init_factors(V, rank, []);        

    %% perform factroization
    
    % new solver

    [~, info_semi] = semi_mu_nmf(V, rank, options);    

    options.use_seminmf_init = false;   

    options.inner_nnls_alg = 'hals';
    [~, info_bcd_hals_semi] = semi_bcd_nmf(V, rank, options);

    options.inner_nnls_alg = 'fpgm';
    [~, info_bcd_fpgm_semi] = semi_bcd_nmf(V, rank, options);

    options.inner_nnls_alg = 'anls_asgroup';
    [~, info_bcd_asgroup_semi] = semi_bcd_nmf(V, rank, options);

    options.inner_nnls_alg = 'anls_bpp';
    [~, info_bcd_bpp_semi] = semi_bcd_nmf(V, rank, options);        


    if ~health_check_mode

        display_graph('epoch','cost', {'Semi-MU', 'Semi-BCD-HALS-Semi', 'Semi-BCD-FPGM-Semi', 'Semi-BCD-ASET-Semi', 'Semi-BCD-BPPM-Semi'}, ...
            [], {info_semi, info_bcd_hals_semi, info_bcd_fpgm_semi, info_bcd_asgroup_semi, info_bcd_bpp_semi});
    
        display_graph('time','cost', {'Semi-MU', 'Semi-BCD-HALS-Semi', 'Semi-BCD-FPGM-Semi', 'Semi-BCD-ASET-Semi', 'Semi-BCD-BPPM-Semi'}, ...
            [], {info_semi, info_bcd_hals_semi, info_bcd_fpgm_semi, info_bcd_asgroup_semi, info_bcd_bpp_semi});             

    end
end