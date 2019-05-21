function comp_nmf_robust_algorithms()
% Test file for NMF with outlier algorithm.
%
% Created by H.Kasai on May 21, 2019

    clc;
    clear;
    close all;


    mu_nmf_flag = 1;
    svrmu_nmf_flag = 1;
    rnmf_flag = 1;
    ronmf_flag = 1;
    rsvrmu_nmf_flag = 1;


    %% generate/load data 
    % d=1: synthetic data in paper, 2: CBCL, 3: ORL, 4: UMISTface
    d = 'CBCL';
    % set the density of outlier
    rho = 1.5;

    fprintf('Loading data ...');
    [N, F, K, Vo, V, Ro] = load_dataset(d, rho);
    fprintf('done\n');
    dim = N * F;



    %% set options
    max_epoch = 100;
    batch_size = floor(N/10); % up to N
    verbose = 2;


    %% set initial data
    %x_init.W = rand(F, K); 
    %x_init.H = rand(K, N);
    %x_init.R = zeros(F, N);
    x_init = [];



    %% execute algorithms
    names = cell(1);
    sols = cell(1);
    infos = cell(1);
    costs = cell(1);
    alg_idx = 0;

    calc_sol = 1;
    % calc solution
    if calc_sol

        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = 0;  

        if 0
            fprintf('Calculating f_opt by NMF ...\n');
            options.alg = 'mu';
            [w_sol, infos_sol] = nmf(V, K, options);
            f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
            fprintf('Done.. f_opt: %.16e\n', f_opt);
        elseif 0
            fprintf('Calculating f_opt by Accelerated NMF ...\n');
            options.alg = 'acc_mu';
            options.alpha = 2;
            options.delta = 0.1;        
            %[w_sol.W,w_sol.H,~, ~] = MUacc(V, x_init.W, x_init.H, 2, 0.01, options.max_epoch,5);
            options.alg = 'hals';
            [w_sol, infos_sol] = nmf(V, K, options);
            [w_sol, infos_sol] = nmf_als(V, K, options);
            f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
            fprintf('Done.. f_opt: %.16e\n', f_opt);
        elseif 1
            fprintf('Calculating f_opt by HALS ...\n');
            options.alg = 'hals';
            [w_sol, infos_sol] = nmf_als(V, K, options);
            f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
            fprintf('Done.. f_opt: %.16e\n', f_opt);
        end

    else
        f_opt = -Inf;
    end



    if mu_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;  
        options.alg = 'mu';

        [x, info] = nmf_mu(V, K, options);

        names{alg_idx} = 'NMF MU'; 
        sols{alg_idx} = x;
        infos{alg_idx} = info;     
        costs{alg_idx} = nmf_cost(Vo, x.W, x.H, zeros(F, N)) * 2 / dim;
    end

    if rnmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;  

        [x, info] = rnmf(V, K, options);

        names{alg_idx} = 'R-NMF'; 
        sols{alg_idx} = x;
        infos{alg_idx} = info;     
        costs{alg_idx} = nmf_cost(Vo, x.W, x.H, zeros(F, N)) * 2 / dim;
    end


    if svrmu_nmf_flag
        alg_idx = alg_idx + 1;
        clear options;
        options.lambda = 0;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;   
        options.f_opt = f_opt;
        options.robust            = false;
        options.x_init_robust     = false;    

        [x, info] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU';
        sols{alg_idx} = x;
        infos{alg_idx} = info;     
        costs{alg_idx} = nmf_cost(Vo, x.W, x.H, zeros(F, N)) * 2 / dim;
    end

    if rsvrmu_nmf_flag
        alg_idx = alg_idx + 1;
        clear options;
        options.lambda = 1;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;   
        options.f_opt = f_opt;
        options.robust            = true;
        options.x_init_robust     = true;    

        [x, info] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'R-SVRMU';
        sols{alg_idx} = x;
        infos{alg_idx} = info;     
        costs{alg_idx} = nmf_cost(Vo, x.W, x.H, zeros(F, N)) * 2 / dim;
    end


    if ronmf_flag
        alg_idx = alg_idx + 1;
        clear options;
        options.lambda = 1;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;   
        options.f_opt = f_opt;
        options.x_init_robust = true;
        options.lambda = 1;

        [x, info] = ronmf(V, K, options);

        names{alg_idx} = 'R-ONMF';
        sols{alg_idx} = x;
        infos{alg_idx} = info;     
        costs{alg_idx} = nmf_cost(Vo, x.W, x.H, zeros(F, N)) * 2 / dim;
    end


    %% plot
    %display_graph('epoch','cost', names, sols, infos);
    %display_graph('grad_calc_count','cost', names, sols, infos);
    display_graph('epoch','optimality_gap', names, sols, infos);
    display_graph('time','optimality_gap', names, sols, infos);
end
