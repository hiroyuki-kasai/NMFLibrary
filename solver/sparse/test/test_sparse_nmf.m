function test_sparse_nmf(varargin)
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 24, 2017


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
        options.verbose = 1;
        options.max_epoch = 100; 
        health_check_mode = false;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end

    % initialize factor matrices
    options.init_alg = 'NNDSVD';
    [options.x_init, ~] = generate_init_factors(V, rank, []);      


    sparse_coeff = 0.5;
    
    
    
    %% perform factroization
    % NMF-MU
    [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);
    
    
    %     [W,H,errs,vout] = nmf_euc_sparse_es(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
    %          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff);     
    
    % Sparse-MU (euc)
    options.lambda = sparse_coeff;
    options.myeps = 1e-20;
    [w_sparse_mu, infos_sparse_mu] = sparse_mu_nmf(V, rank, options);
    
    
    %     [W,H,errs,vout] = nmf_kl_sparse_es(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
    %          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff); 
    
    % Sparse-MU (kl-div)
    options.lambda = sparse_coeff;
    options.myeps = 1e-20;
    options.metric_type = 'kl-div';
    [w_sparse_mu_kl, infos_sparse_mu_kl] = sparse_mu_nmf(V, rank, options); 
    
    
    %     [W,H,errs,vout] = nmf_kl_sparse_v(V, rank, 'verb', 3, 'niter', options.max_epoch, ...
    %          'W0', x_init.W, 'H0', x_init.H, 'alpha', sparse_coeff);     
    
    
    % Sparse-MU-V (kl-div)
    options.lambda = sparse_coeff;
    options.myeps = 1e-20;
    options.metric_type = 'kl-div';
    [w_sparse_mu_kl_v, infos_sparse_mu_kl_v] = sparse_mu_v_nmf(V, rank, options);     
    
    
    
    % Nonsmooth-NMF
    options.theta = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost.type = 'euc'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_nsnmf, infos_nsnmf] = ns_nmf(V, rank, options); 
    
    % SparseNMF
    options.lambda = 1000; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost = 'euc'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_sparsenmf, infos_sparsenmf] = sparse_nmf(V, rank, options);  
    
    % NMF with sparse constrained
    options.lambda = 1; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    options.cost = 'euc'; %- what cost function to use; 'eucl' (default) or 'kl'
    [w_nmfsc, infos_nmfsc] = sc_nmf(V, rank, options);       

    % Projective Sparse NMF
    [w_proj_sparse_nmf1, infos_proj_sparse_nmf1] = proj_sparse_nmf(V, rank, options);
    
    options.sW = 0.8;
    [w_proj_sparse_nmf2, infos_proj_sparse_nmf2] = proj_sparse_nmf(V, rank, options);   
    
    options.sH = 0.8;     
    options.FPGM = true;    
    [w_proj_sparse_nmf3, infos_proj_sparse_nmf3] = proj_sparse_nmf(V, rank, options); 

    options.lambda = 0;
    options.eta = 0.5;    
    [w_palm_sparse_smooth, info_palm_sparse_smoot] = palm_sparse_smooth_nmf(V, rank, options);    
    
    
    if ~health_check_mode      
        
        %% plot
        display_graph('epoch','cost',{'Fro-MU','Sparse-MU','Sparse-MU-kl-div','Sparse-MU-V',...
            'NS-NMF','Sparse-NMF','NMF-SC','Proj-Sparse-NMF (std)','Proj-Sparse-NMF (W)',...
            'Proj-Sparse-NMF (WH:FPGM)','PALM-Sparse-Smooth'}, ...
            {w_nmf_mu, w_sparse_mu,w_sparse_mu_kl,w_sparse_mu_kl_v,w_nsnmf,w_sparsenmf,...
            w_nmfsc,w_proj_sparse_nmf1,w_proj_sparse_nmf2,w_proj_sparse_nmf3,w_palm_sparse_smooth}, ...
            {infos_nmf_mu,infos_sparse_mu,infos_sparse_mu_kl,infos_sparse_mu_kl_v,...
            infos_nsnmf,infos_sparsenmf,infos_nmfsc,infos_proj_sparse_nmf1,...
            infos_proj_sparse_nmf2,infos_proj_sparse_nmf3,info_palm_sparse_smoot});
        
        
        % sparseness 
        %   defined by
        %       Patrik O. Hoyer, 
        %       "Non-negative matrix factorization with sparseness constraints," 
        %       Journal of Machine Learning Research, vol.5, pp.1457-1469, 2004.
        
        display_sparsity_graph({'Fro-MU','Sparse-MU','Sparse-MU-kl-div','Sparse-MU-V',...
            'NS-NMF','Sparse-NMF','NMF-SC','Proj-Sparse-NMF','PALM-Sparse-Smooth'},...
            {w_nmf_mu, w_sparse_mu,w_sparse_mu_kl,w_sparse_mu_kl_v,w_nsnmf,w_sparsenmf,...
            w_nmfsc,w_proj_sparse_nmf1,w_proj_sparse_nmf2,w_proj_sparse_nmf3,w_palm_sparse_smooth});  

    end


end