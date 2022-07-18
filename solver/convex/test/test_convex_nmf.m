function test_convex_nmf(varargin)
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 30, 2022

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
    H = rand(rank, size(V, 2));
    W = H'*diag(1./sum(H, 2)');

    options.x_init.W = W;
    options.x_init.H = H;    


    [~, info_std] = convex_mu_nmf(V, rank, options);    

    options.sub_mode = 'kernel';
    [~, info_kernel_rbf] = convex_mu_nmf(V, rank, options);       

    options.kernel = 'polynomial';    
    [~, info_kernel_poly] = convex_mu_nmf(V, rank, options); 

    options.kernel = 'linear';    
    [~, info_kernel_lin] = convex_mu_nmf(V, rank, options);  

    options.kernel = 'sigmoid';    
    [~, info_kernel_sig] = convex_mu_nmf(V, rank, options);      
 
    
    %% plot
    if ~health_check_mode      
        display_graph('epoch','cost', {'Standard', 'Kernel (rbf)', 'Kernel (polynomial)', 'Kernel (linear)', 'Kernel (sigmoid)'}, ...
            [], {info_std, info_kernel_rbf, info_kernel_poly, info_kernel_lin, info_kernel_sig});
        display_graph('time','cost', {'Standard', 'Kernel (rbf)', 'Kernel (polynomial)', 'Kernel (linear)', 'Kernel (sigmoid)'}, ...
            [], {info_std, info_kernel_rbf, info_kernel_poly, info_kernel_lin, info_kernel_sig});    
    end
    
end