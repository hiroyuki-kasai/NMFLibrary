function test_seminmf()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 21, 2017

    clc;
    clear;
    close all;

    %% generate synthetic data of (mxn) matrix       
    m = 500;
    n = 100;
    V = rand(m,n);
    
    
    %% Initialize of rank to be factorized
    rank = 5;

    
    %% Initialize of W and H
    options = [];
    [x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
    options.x_init = x_init;
    options.verbose = 2;
    
    

    %% perform factroization
    % NMF-MU
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
    
    % Semi-NMF
    [w_seminmf, infos_seminmf] = semi_nmf(V, rank, options);
    
    % nsNMF
    options.theta = 0; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_nsnmf, infos_nsnmf] = ns_nmf(V, rank, options); 
    
    % SparseNMF
    options.lambda = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_sparsenmf, infos_sparsenmf] = sparse_nmf(V, rank, options);  
    
    % SparseNMF
    options.lambda = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost = 'EUC'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_nmfsc, infos_nmfsc] = nmf_sc(V, rank, options);       
    
   
    % Ne-NMF
    %[W,H,it,ela,HIS]=NeNMF(V,reducedDim,'TYPE','L1R');
    %[W,H,it,ela,HIS]=NeNMF(V,reducedDim,'TYPE','L2R');
    %[W,H,it,ela,HIS]=NeNMF(V,reducedDim,'TYPE','MR','S_MTX',S);
    %[w_nenmf, infos_nenmf] = NeNMF(V, rank);    
    
    % Hierarchical ALS
    options.alg = 'hals';
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);       
    
    
    %% plot
    display_graph('epoch','cost', {'NMF-MU', 'Semi-NMF', 'nsNMF', 'Sparse NMF', 'NMFsc', 'HALS'}, {w_nmf_mu, w_seminmf, w_nsnmf, w_sparsenmf, w_nmfsc, w_nmf_hals}, {infos_nmf_mu, infos_seminmf, infos_nsnmf, infos_sparsenmf, infos_nmfsc, infos_nmf_hals});
    display_graph('time','cost', {'NMF-MU','Semi-NMF', 'nsNMF', 'Sparse NMF', 'NMFsc', 'HALS'}, {w_nmf_mu, w_seminmf, w_nsnmf, w_sparsenmf, w_nmfsc, w_nmf_hals}, {infos_nmf_mu, infos_seminmf, infos_nsnmf, infos_sparsenmf, infos_nmfsc, infos_nmf_hals});
    
end


