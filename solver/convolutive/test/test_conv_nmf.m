function test_conv_nmf(varargin)
%
% demonstration file for NMFLibrary.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 08, 2022

    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 300;
        n = 500;
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
    T = 5;

    %% Initialize factor matrices    
    [m, n] = size(V);
    W_groundtruth = abs(randn(m, rank, T));
    H_groundtruth = abs(randn(rank, n));
    
    % generate data to approximate
    for t=0:T-1
        tW = W_groundtruth(:,:,t+1);
        tH = shift_t(H_groundtruth,t);
        V = V + tW * tH;
    end
    
    % the initialization for all the tested algorithms
    W = abs(randn(m, rank, T));
    H = abs(randn(rank, n));
    
    % Compute V_hat
    V_hat = zeros(m, n);
    for t=0:T-1
        V_hat = V_hat + W(:,:,t+1)*shift_t(H,t);
    end

    %% perform algorithms
    options.x_init.W = W;
    options.x_init.H = H;
    options.metric_type = 'beta-div';
    %options.metric_type = 'euc';  
    %options.metric_type = 'kl-div';     
    %options.metric_type = 'is-div';      
    options.d_beta = 2;     

    [~, infos_heur_mu] = heuristic_mu_conv_nmf(V, rank, T, options);    
    [~, infos_mu1] = mu_conv_nmf(V, rank, T, options);
    options.alg = 'type2';
    [~, infos_mu2] = mu_conv_nmf(V, rank, T, options); 

    [~, infos_admm_seq] = admm_seq_conv_nmf(V, rank, T, options);

    [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);    

    if strcmp(options.metric_type, 'beta-div')
        if options.d_beta == 0
            options.metric_type = 'is-div'; 
        elseif  options.d_beta == 1
            options.metric_type = 'kl-div'; 
        elseif options.d_beta == 2
         options.metric_type = 'euc';
        end
        [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);        
    end


    if ~health_check_mode    
        display_graph('epoch','cost', {'Heur-MU', 'MU1', 'MU2', 'ADMM-Seq', 'ADMM-Y'}, [], ...
            {infos_heur_mu, infos_mu1, infos_mu2, infos_admm_seq, infos_admm_y});    
    
        display_graph('time','cost', {'Heur-MU', 'MU1', 'MU2', 'ADMM-Seq', 'ADMM-Y'}, [], ...
            {infos_heur_mu, infos_mu1, infos_mu2, infos_admm_seq, infos_admm_y});
    else

        options.metric_type = 'euc';   
        options.d_beta = 2;     
    
        [~, infos_heur_mu] = heuristic_mu_conv_nmf(V, rank, T, options);    
        [~, infos_mu1] = mu_conv_nmf(V, rank, T, options);
        options.alg = 'type2';
        [~, infos_mu2] = mu_conv_nmf(V, rank, T, options); 
    
        [~, infos_admm_seq] = admm_seq_conv_nmf(V, rank, T, options);
    
        [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);    
    
        if strcmp(options.metric_type, 'beta-div')
            if options.d_beta == 0
                options.metric_type = 'is-div'; 
            elseif  options.d_beta == 1
                options.metric_type = 'kl-div'; 
            elseif options.d_beta == 2
                options.metric_type = 'euc';
            end
            [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);        
        end

        options.metric_type = 'kl-div';      
        options.d_beta = 2;     
    
        [~, infos_heur_mu] = heuristic_mu_conv_nmf(V, rank, T, options);    
        [~, infos_mu1] = mu_conv_nmf(V, rank, T, options);
        options.alg = 'type2';
        [~, infos_mu2] = mu_conv_nmf(V, rank, T, options); 
    
        [~, infos_admm_seq] = admm_seq_conv_nmf(V, rank, T, options);
    
        [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);    
    
        if strcmp(options.metric_type, 'beta-div')
            if options.d_beta == 0
                options.metric_type = 'is-div'; 
            elseif  options.d_beta == 1
                options.metric_type = 'kl-div'; 
            elseif options.d_beta == 2
                options.metric_type = 'euc';
            end
            [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);        
        end        

        options.metric_type = 'is-div';    
        options.d_beta = 0;     
    
        [~, infos_heur_mu] = heuristic_mu_conv_nmf(V, rank, T, options);    
        [~, infos_mu1] = mu_conv_nmf(V, rank, T, options);
        options.alg = 'type2';
        [~, infos_mu2] = mu_conv_nmf(V, rank, T, options); 
    
        [~, infos_admm_seq] = admm_seq_conv_nmf(V, rank, T, options);
    
        [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);    
    
        if strcmp(options.metric_type, 'beta-div')
            if options.d_beta == 0
                options.metric_type = 'is-div'; 
            elseif  options.d_beta == 1
                options.metric_type = 'kl-div'; 
            elseif options.d_beta == 2
                options.metric_type = 'euc';
            end
            [~, infos_admm_y] = admm_y_conv_nmf(V, rank, T, options);        
        end         
    end



end
             