function test_prob_nmf(varargin)
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on May 20, 2019
%


    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')
    
        m = 500;
        n = 100;
        V = rand(m,n);
        rank = 5;
        options = [];
        options.verbose = 2;
        options.max_epoch = 100;
        calc_sol = 1;


        %% load CBCL face datasets
        % fprintf('Loading data ...');
        %V = importdata('../../../data/CBCL_face.mat');
        V = importdata('../../../data/R.txt');
       
        health_check_mode = false;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};

        calc_sol = 0;
        health_check_mode = true;
    end

    [m, n] = size(V);
    Vo = V;
    dim = m * n; 

    pnmf_vb_flag = 1;
    pnmf_vb_ard_flag = 1;
    nmf_mu_flag = 1;
    nmf_hals_flag = 1;
    prob_nmf_flag = 1;
    
    
    %% calculate optimal solution

    if calc_sol
        clear options_sol;
        options_sol.verbose = 0;  
        options_sol.max_epoch = 10000;
        
        if 1
            fprintf('Calculating f_opt by HALS ...\n');
            options_sol.alg = 'hals';
            [w_sol, infos_sol] = als_nmf(V, rank, options_sol);
            f_opt = nmf_cost(V, w_sol.W, w_sol.H, []);
            
        else
            fprintf('Calculating f_opt by ANLS ...\n');
            options_sol.alg = 'anls_asgroup';
            options_sol.alg = 'anls_asgivens';
            options_sol.alg = 'anls_bpp';
            [w_sol, infos_sol] = anls_nmf(V, rank, options_sol);
            f_opt = nmf_cost(V, w_sol.W, w_sol.H, []);
        end
        fprintf('Done.. f_opt: %.16e\n', f_opt);

        options.f_opt = f_opt;
    end
    
    
    %% execute algorithms
    names = cell(1);
    sols = cell(1);
    infos = cell(1);
    costs = cell(1);
    alg_idx = 0;


    if prob_nmf_flag
        alg_idx = alg_idx + 1;  

        options.tol_optgap = -Inf;
        
        [w_nmf_prob, infos_nmf_prob] = prob_nmf(V, rank, options);
        
        names{alg_idx} = 'Prob-NMF'; 
        sols{alg_idx} = w_nmf_prob;
        infos{alg_idx} = infos_nmf_prob;     
        costs{alg_idx} = nmf_cost(Vo, w_nmf_prob.W, w_nmf_prob.H, []) * 2 / dim;
    end    
    
    
    if pnmf_vb_flag
        alg_idx = alg_idx + 1;  
        
        %
        lambdaU = 0.1;
        lambdaV = 0.1;
        alphatau = 1;
        betatau = 1;
        alpha0 = 1;
        beta0 = 1;
        options.hyperparams.alphatau = alphatau;
        options.hyperparams.betatau = betatau;
        options.hyperparams.alpha0 = alpha0;
        options.hyperparams.beta0 = beta0;
        options.hyperparams.lambdaU = lambdaU;
        options.hyperparams.lambdaV = lambdaV;
        options.ard = false;    
        options.init_alg = 'prob_expectation';
        %options.init_alg = 'prob_random';
        
        [w_pnmf_vb, infos_pnmf_vb] = vb_pro_nmf(V, rank, options);
        
        names{alg_idx} = 'VB-Pro-NMF'; 
        sols{alg_idx} = w_pnmf_vb;
        infos{alg_idx} = infos_pnmf_vb;     
        costs{alg_idx} = nmf_cost(Vo, w_pnmf_vb.W, w_pnmf_vb.H, []) * 2 / dim;
    end
    
    if pnmf_vb_ard_flag
        alg_idx = alg_idx + 1;  
        
        %
        lambdaU = 0.1;
        lambdaV = 0.1;
        alphatau = 1;
        betatau = 1;
        alpha0 = 1;
        beta0 = 1;
        options.hyperparams.alphatau = alphatau;
        options.hyperparams.betatau = betatau;
        options.hyperparams.alpha0 = alpha0;
        options.hyperparams.beta0 = beta0;
        options.hyperparams.lambdaU = lambdaU;
        options.hyperparams.lambdaV = lambdaV;
        options.ard = true;    
        options.init_alg = 'prob_expectation';
        %options.init_alg = 'prob_random';
        
        [w_pnmf_vb, infos_pnmf_vb] = vb_pro_nmf(V, rank, options);
        
        names{alg_idx} = 'VB-Pro-NMF (ARD)'; 
        sols{alg_idx} = w_pnmf_vb;
        infos{alg_idx} = infos_pnmf_vb;     
        costs{alg_idx} = nmf_cost(Vo, w_pnmf_vb.W, w_pnmf_vb.H, []) * 2 / dim;
    end
    
    
    if nmf_mu_flag
        alg_idx = alg_idx + 1;  

        options.alg = 'mu';
        options.init_alg = 'NNDSVD';
        
        [w_nmf_mu, infos_nmf_mu] = fro_mu_nmf(V, rank, options);
        
        names{alg_idx} = 'Fro-MU'; 
        sols{alg_idx} = w_nmf_mu;
        infos{alg_idx} = infos_nmf_mu;     
        costs{alg_idx} = nmf_cost(Vo, w_nmf_mu.W, w_nmf_mu.H, []) * 2 / dim;
    end
        
    
    if nmf_hals_flag
        alg_idx = alg_idx + 1;  
      
        options.alg = 'hals';
        
        [w_nmf_hals, infos_nmf_hals] = als_nmf(V, rank, options);
        
        names{alg_idx} = 'HALS'; 
        sols{alg_idx} = w_nmf_hals;
        infos{alg_idx} = infos_nmf_hals;     
        costs{alg_idx} = nmf_cost(Vo, w_nmf_hals.W, w_nmf_hals.H, []) * 2 / dim;
    end
    
    
    if ~health_check_mode      
        %% plot
        display_graph('epoch','cost', names, sols, infos);
        display_graph('time','cost', names, sols, infos);
        display_graph('epoch','optimality_gap', names, sols, infos);
        display_graph('time','optimality_gap', names, sols, infos);
    end

end