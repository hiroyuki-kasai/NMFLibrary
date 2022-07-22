function test_online_nmf(varargin)
% Test file for stochastic/online NMF.
%
% Created by H.Kasai on Apr. 17, 2018


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
        max_epoch = 100;
        options.max_epoch = max_epoch; 
        health_check_mode = false;
        calc_sol = 1;
        outlier_rho = 0.0;
        lambda = 1;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};
        max_epoch = options.max_epoch;
        calc_sol = 0;
        health_check_mode = true;
    end

    max_h_repeat    = 5;
    svrmu_inneriter = 5;

    % initialize factor matrices
    [x_init, ~] = generate_init_factors(V, rank, []); 
    options.x_init = x_init;      



    %% calc solution
    if calc_sol

        clear options_sol;
        options_sol.max_epoch = max_epoch*100;
        options_sol.x_init = x_init;
        options_sol.verbose = 0;  
        options_sol.verbose = 1;        

        if ~outlier_rho
            fprintf('Calculating f_opt by HALS ...\n');
            options_sol.alg = 'hals';
            [w_sol, infos_sol] = als_nmf(V, rank, options_sol);
        else
            fprintf('Calculating f_opt by R-NMF ...\n');        
            options_sol.lambda = lambda;
            [w_sol, infos_sol] = robust_mu_nmf(V, rank, options_sol);
        end
        
        f_opt = infos_sol.cost(end);
        fprintf('Done.. f_opt: %.16e\n', f_opt);        
    else
        f_opt = -Inf;
    end
    
    
    
   

    
    %% online_mu_nmf
    [w_onmf, infos_onmf] = online_mu_nmf(V, rank, options);

    %% acc_online_mu_nmf    
    options.rep_mode = 'fix';
    options.w_repeat = 1;     
    options.h_repeat = max_h_repeat; 
    [w_onmf_acc, infos_onmf_acc] = acc_online_mu_nmf(V, rank, options);

    options.rep_mode = 'adaptive';    
    options.w_repeat = 1;     
    options.h_repeat = max_h_repeat; 
    [w_onmf_acc, infos_onmf_acc] = acc_online_mu_nmf(V, rank, options);    


    %% inf
    [w_inmf, infos_inmf] = incremental_mu_nmf(V, rank, options);

    options.online = 1; % online mode
    [w_online_inmf, infos_online_inmf] = incremental_mu_nmf(V, rank, options);


    %% asag_mu
    [w_asag_mu_nmf, infos_asag_mu_nmf] = asag_mu_nmf(V, rank, options);

    %% smu
    [w_smu_nmf, infos_smu_nmf] = smu_nmf(V, rank, options);

    options.accel = true;
    options.rep_mode = 'fix';
    options.w_repeat = 1;
    options.h_repeat = max_h_repeat;   
    [w_smu_nmf_acc, infos_smu_nmf_acc] = smu_nmf(V, rank, options);
    options.accel = false;

    options.ls = false;
    options.rep_mode = 'adaptive';
    options.w_repeat = 1;
    options.h_repeat = max_h_repeat;   
    [w_smu_nmf_acc, infos_smu_nmf_acc] = smu_nmf(V, rank, options);
    options.accel = 0; 
    options.ls = 0;

    %% spg
    [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, rank, options);

    options.accel = 1; 
    options.h_repeat = max_h_repeat; 
    [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, rank, options);


    options.accel = 0; 
    %options.h_repeat = max_h_repeat; 
    options.W_sub_mode = 'Nesterov';
    [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, rank, options);
 

    options.accel = 0; 
    options.ls = 1;
    [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, rank, options);

    
    %% svrmu
    options.x_init.R = zeros(size(V)); % enfore non-robust mode          
    options.fast_calc = 0;
    options.repeat_inneriter = svrmu_inneriter;
    options.accel = 0;  
    options.ls = 0;
    options.ratio = 1; % 1: original 1<: adaptive
    options.robust = false;
    options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    
    [w_svrmu_nmf, infos_svrmu_nmf] = svrmu_nmf(V, rank, options);


    options.accel = 1;
    options.rep_mode = 'fix';
    [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, rank, options);

    options.rep_mode = 'adaptive';
    options.accel = 0;    
    [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, rank, options);    

    options.accel = 1;
    [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, rank, options);


  
    options.fast_calc = 0;
    options.accel = 1;
    options.rep_mode = 'adaptive';
    options.ratio = 0.3; % 1: original 1<: adaptive
    options.robust = false;        
    [w_svrmu_acc_adaptive_nmf, infos_svrmu_acc_adaptive_nmf] = svrmu_nmf(V, rank, options);




end