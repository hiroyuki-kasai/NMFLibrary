function comp_nmf_online_algorithms_test()
% Test file for stochastic/online NMF.
%
% Created by H.Kasai on Apr. 17, 2018

    clc;
    clear;
    close all;

    %% set options
    max_epoch                       = 100;
    verbose                         = 2; 
    batch_size                      = 50;
    permute_on                      = true;
    svrmu_inneriter                 = 5;        
    max_h_repeat                    = 5;
    calc_sol                        = 1;  
    lambda                          = 1;    
    
    
    dataset                         = 'CBCL';
    %dataset                         = 'COIL';
    %dataset                         = 'Synthetic';
    
    F                               = 100;
    K                               = 10;
    N                               = 500;
    noise_level                     = 0.0;
    outlier_rho                     = 0.0; % 0.9
    

    dic_display_flag                = 1;
    plot_flag                       = 1;
    clustering_flag                 = 1;
    denoise_display_flag            = 1;



    %% algorithms
    if outlier_rho == 0
        % batch
        nmf_hals_flag                   = 0;
        nmf_acc_hals_flag               = 0;
        rnmf_flag                       = 0;
    
        % spg
        spg_nmf_flag                    = 0;
        spg_acc_nmf_flag                = 0;  
        apg_nmf_flag                    = 0; 
        spg_ls_nmf_flag                 = 0;        
        spg_precon_ls_nmf_flag          = 0;  
        
        % online mu
        ronmf_flag                      = 0;
        onmf_flag                       = 0;
        onmf_acc_flag                   = 0;
        inmf_flag                       = 0;
        inmf_online_flag                = 0;
        asag_mu_nmf_flag                = 0;        
        
        % smu
        smu_nmf_flag                    = 0;
        smu_acc_nmf_flag                = 0;
        smu_ls_nmf_flag                 = 0;

        % svrmu
        svrmu_nmf_flag                  = 0;
        svrmu_acc_nmf_flag              = 1;
        svrmu_ls_nmf_flag               = 0;
        svrmu_ls_acc_nmf_flag           = 0;
        svrmu_precon_ls_nmf_flag        = 0;
        svrmu_precon_ls_acc_nmf_flag    = 0;    
        rsvrmu_acc_nmf_flag             = 0;
        svrmu_acc_adaptive_nmf_flag     = 0;
    else
        % batch
        nmf_hals_flag                   = 0;
        nmf_acc_hals_flag               = 0;
        rnmf_flag                       = 1;
        
        % spg
        spg_nmf_flag                    = 0;
        spg_acc_nmf_flag                = 0;  
        apg_nmf_flag                    = 0; 
        spg_ls_nmf_flag                 = 0;        
        spg_precon_ls_nmf_flag          = 0;  
        
        % online mu
        ronmf_flag                      = 1;
        onmf_flag                       = 0;
        onmf_acc_flag                   = 0;
        inmf_flag                       = 0;
        inmf_online_flag                = 0;
        asag_mu_nmf_flag                = 0;        
        
        % smu
        smu_nmf_flag                    = 0;
        smu_acc_nmf_flag                = 0;
        smu_ls_nmf_flag                 = 0;

        % svrmu
        svrmu_nmf_flag                  = 0;
        svrmu_acc_nmf_flag              = 0;
        svrmu_ls_nmf_flag               = 0;
        svrmu_ls_acc_nmf_flag           = 0;
        svrmu_precon_ls_nmf_flag        = 0;
        svrmu_precon_ls_acc_nmf_flag    = 0;    
        rsvrmu_acc_nmf_flag             = 1;
        svrmu_acc_adaptive_nmf_flag     = 0;
    end
    
    
    
    %% generate/load data
    fprintf('Loading data [%s] ...', dataset);
    if strcmp(dataset, 'PIE') || strcmp(dataset, 'AR') || strcmp(dataset, 'COIL') || strcmp(dataset, 'CBCL')
        
        if strcmp(dataset, 'PIE')
            img_in_dim  = [32, 32];
            img_out_dim = img_in_dim; 
            batch_size  = 100; % up to N 
            dic_display_dim = [4, 5]; 

        elseif strcmp(dataset, 'AR')
            img_in_dim  = [28, 20];
            img_out_dim = [28, 28];
            batch_size  = 100; % up to N 
            dic_display_dim = [10, 12]; 
           
        elseif strcmp(dataset, 'COIL')
            img_in_dim  = [32, 32];
            img_out_dim = img_in_dim;
            batch_size  = 100; % up to N
            dic_display_dim = [4, 5];  

        elseif strcmp(dataset, 'CBCL')
            img_in_dim  = [19, 19];
            img_out_dim = img_in_dim;
            batch_size  = 100; % up to N
            dic_display_dim = [7, 7];  
              
        end
        
        options = [];
        [N, F, K, Vo, V, ~, classes] = load_dataset(dataset, outlier_rho);    
        
        oriimg_display_flag = 1;
        
        if isempty(classes)
            clustering_flag = 0;
        end

    elseif strcmp(dataset, 'Synthetic')
        [Vo, Ro, No, V] = generate_syntheticdata(F, N, K, noise_level, outlier_rho);
        
        oriimg_display_flag     = 0;
        clustering_flag         = 0;
        denoise_display_flag    = 0;
        dic_display_flag        = 0;
    end
    
    dim = F * N;
    fprintf(' done\n');


    %% set initial data
    if 0
        x_init.W = rand(F, K); 
        x_init.H = rand(K, N);
    elseif 1
        [W_init, H_init] = Qiao_SVD_Init(V, K);
        x_init.W = W_init;
        x_init.H = H_init;
    else 
        [W_init, H_init] = NNDSVD(V, K, 0);
        x_init.W = W_init;
        x_init.H = H_init;        
    end
    
    if outlier_rho == 0
        x_init.R = zeros(F, N);
    else
        x_init.R = rand(F, N);
    end    


    %% calc solution
    if calc_sol

        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = 0;  
        options.verbose = verbose;        

        if ~outlier_rho
            fprintf('Calculating f_opt by HALS ...\n');
            options.alg = 'hals';
            [w_sol, infos_sol] = nmf_als(V, K, options);
        else
            fprintf('Calculating f_opt by R-NMF ...\n');        
            options.lambda = lambda;
            [w_sol, infos_sol] = rnmf(V, K, options);
        end
        
        f_opt = infos_sol.cost(end);
        fprintf('Done.. f_opt: %.16e\n', f_opt);        
    else
        f_opt = -Inf;
    end
    
    
    
    %% execute algorithms
    names = cell(1);
    sols = cell(1);
    infos = cell(1);
    costs = cell(1);
    alg_idx = 0;

    
    %% Batch
    if nmf_hals_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;    
        options.alg = 'hals';

        [w_nmf_hals, infos_nmf_hals] = nmf_als(V, K, options);

        names{alg_idx} = 'NMF HALS'; 
        sols{alg_idx} = w_nmf_hals;
        infos{alg_idx} = infos_nmf_hals;     
        costs{alg_idx} = nmf_cost(Vo, w_nmf_hals.W, w_nmf_hals.H, zeros(F, N)) * 2 / dim;
    end

    if nmf_acc_hals_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;    
        options.alg = 'acc_hals';


        [w_nmf_acc_hals, infos_nmf_acc_hals] = nmf_als(V, K, options);

        names{alg_idx} = 'NMF ACC HALS'; 
        sols{alg_idx} = w_nmf_acc_hals;
        infos{alg_idx} = infos_nmf_acc_hals;     
        costs{alg_idx} = nmf_cost(Vo, w_nmf_acc_hals.W, w_nmf_acc_hals.H, zeros(F, N)) * 2 / dim;
    end
    
    if rnmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.max_epoch = max_epoch;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;    

        [w_rnmf, infos_rnmf] = rnmf(V, K, options);

        names{alg_idx} = 'R-NMF(Batch)'; 
        sols{alg_idx} = w_rnmf;
        infos{alg_idx} = infos_rnmf;     
        costs{alg_idx} = nmf_cost(Vo, w_rnmf.W, w_rnmf.H, zeros(F, N)) * 2 / dim;
    end 
    
    
    
    %% Online MU
    if ronmf_flag
        alg_idx = alg_idx + 1;
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;   
        options.f_opt = f_opt;

        [w_ronmf, infos_ronmf] = ronmf(V, K, options);

        names{alg_idx} = 'R-ONMF';
        sols{alg_idx} = w_ronmf;
        infos{alg_idx} = infos_ronmf;   
        costs{alg_idx} = nmf_cost(Vo, w_ronmf.W, w_ronmf.H, zeros(F, N)) * 2 / dim;
    end

    if onmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        %options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;

        [w_onmf, infos_onmf] = onmf(V, K, options);

        names{alg_idx} = 'ONMF';   
        sols{alg_idx} = w_onmf;
        infos{alg_idx} = infos_onmf;     
        costs{alg_idx} = nmf_cost(Vo, w_onmf.W, w_onmf.H, zeros(F, N)) * 2 / dim;
    end

    if onmf_acc_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';    
        options.w_repeat = 1;     
        options.h_repeat = max_h_repeat; 

        [w_onmf_acc, infos_onmf_acc] = onmf_acc(V, K, options);

        names{alg_idx} = 'ONMF-ACC';   
        sols{alg_idx} = w_onmf_acc;
        infos{alg_idx} = infos_onmf_acc;     
        costs{alg_idx} = nmf_cost(Vo, w_onmf_acc.W, w_onmf_acc.H, zeros(F, N)) * 2 / dim;
    end

    if inmf_flag
        alg_idx = alg_idx + 1; 
        clear options;
        options.max_epoch = max_epoch;
        options.online = 0; % online mode
        options.x_init = x_init;
        options.verbose = 1;
        options.f_opt = f_opt;  
        options.batch_size = batch_size;  

        [w_inmf, infos_inmf] = inmf(V, K, options);

        names{alg_idx} = 'INMF'; 
        sols{alg_idx} = w_inmf;
        infos{alg_idx} = infos_inmf; 
        costs{alg_idx} = nmf_cost(Vo, w_inmf.W, w_inmf.H, zeros(F, N)) * 2 / dim;
    end

    if inmf_online_flag
        alg_idx = alg_idx + 1; 
        clear options;
        options.max_epoch = 1;
        options.online = 1; % online mode
        options.max_inneriter = 100;
        options.x_init = x_init;
        options.verbose = 2;
        options.f_opt = f_opt;  
        options.batch_size = batch_size;  

        [w_online_inmf, infos_online_inmf] = inmf(V, K, options);

        names{alg_idx} = 'INMF (Online)'; 
        sols{alg_idx} = w_online_inmf;
        infos{alg_idx} = infos_online_inmf; 
        costs{alg_idx} = nmf_cost(Vo, w_online_inmf.W, w_online_inmf.H, zeros(F, N)) * 2 / dim;
    end
    
    %%
    if asag_mu_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = 1;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.permute_on = permute_on;
        options.lambda = 0.5;
        options.f_opt = f_opt;    

        [w_asag_mu_nmf, infos_asag_mu_nmf] = asag_mu_nmf(V, K, options);

        names{alg_idx} = 'ASAG-MU';   
        sols{alg_idx} = w_asag_mu_nmf;
        infos{alg_idx} = infos_asag_mu_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_asag_mu_nmf.W, w_asag_mu_nmf.H, zeros(F, N)) * 2 / dim;
    end    

    
    
    
    %% SMU
    if smu_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 0; 

        [w_smu_nmf, infos_smu_nmf] = smu_nmf(V, K, options);

        names{alg_idx} = 'SMU';   
        sols{alg_idx} = w_smu_nmf;
        infos{alg_idx} = infos_smu_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_smu_nmf.W, w_smu_nmf.H, zeros(F, N)) * 2 / dim;
    end

    if smu_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat;   

        [w_smu_nmf_acc, infos_smu_nmf_acc] = smu_nmf(V, K, options);

        names{alg_idx} = 'SMU-ACC';   
        sols{alg_idx} = w_smu_nmf_acc;
        infos{alg_idx} = infos_smu_nmf_acc;     
        costs{alg_idx} = nmf_cost(Vo, w_smu_nmf_acc.W, w_smu_nmf_acc.H, zeros(F, N)) * 2 / dim;
    end
    
    if smu_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.ls = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat;   

        [w_smu_nmf_acc, infos_smu_nmf_acc] = smu_nmf(V, K, options);

        names{alg_idx} = 'SMU-LS';   
        sols{alg_idx} = w_smu_nmf_acc;
        infos{alg_idx} = infos_smu_nmf_acc;     
        costs{alg_idx} = nmf_cost(Vo, w_smu_nmf_acc.W, w_smu_nmf_acc.H, zeros(F, N)) * 2 / dim;
    end   
    
    
    if smu_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.ls = 1;
        options.precon = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat;   

        [w_smu_nmf_acc, infos_smu_nmf_acc] = smu_nmf(V, K, options);

        names{alg_idx} = 'SMU-Precon-LS';   
        sols{alg_idx} = w_smu_nmf_acc;
        infos{alg_idx} = infos_smu_nmf_acc;     
        costs{alg_idx} = nmf_cost(Vo, w_smu_nmf_acc.W, w_smu_nmf_acc.H, zeros(F, N)) * 2 / dim;
    end     


    %% SPG
    if spg_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 0; 
        options.ls = 0;

        [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, K, options);

        names{alg_idx} = 'SPG';   
        sols{alg_idx} = w_spg_nmf;
        infos{alg_idx} = infos_spg_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_spg_nmf.W, w_spg_nmf.H, zeros(F, N)) * 2 / dim;
    end  
    
    if spg_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 1; 
        options.ls = 0;
        options.h_repeat = max_h_repeat; 

        [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, K, options);

        names{alg_idx} = 'SPG-ACC';   
        sols{alg_idx} = w_spg_nmf;
        infos{alg_idx} = infos_spg_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_spg_nmf.W, w_spg_nmf.H, zeros(F, N)) * 2 / dim;
    end   
    
    
    if apg_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 0; 
        options.ls = 0;
        %options.h_repeat = max_h_repeat; 
        options.W_sub_mode = 'Nesterov';

        [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, K, options);

        names{alg_idx} = 'APG';   
        sols{alg_idx} = w_spg_nmf;
        infos{alg_idx} = infos_spg_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_spg_nmf.W, w_spg_nmf.H, zeros(F, N)) * 2 / dim;
    end         
    
    
    
    if spg_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 0; 
        options.ls = 1;

        [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, K, options);

        names{alg_idx} = 'SPG-LS';   
        sols{alg_idx} = w_spg_nmf;
        infos{alg_idx} = infos_spg_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_spg_nmf.W, w_spg_nmf.H, zeros(F, N)) * 2 / dim;
    end     
    

    if spg_precon_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;
        options.max_epoch = max_epoch;
        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose; 
        options.f_opt = f_opt;
        options.accel = 0; 
        options.ls = 1;
        options.precon = 1;

        [w_spg_nmf, infos_spg_nmf] = spg_nmf(V, K, options);

        names{alg_idx} = 'SPG-Precon-LS';   
        sols{alg_idx} = w_spg_nmf;
        infos{alg_idx} = infos_spg_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_spg_nmf.W, w_spg_nmf.H, zeros(F, N)) * 2 / dim;
    end 
        


    %% SVRMU
    if svrmu_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.x_init.R = zeros(F, N); % enfore non-robust mode          
        options.verbose = verbose;  
        options.permute_on = permute_on;
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 0;  
        options.ratio = 1; % 1: original 1<: adaptive
        options.permute_on = permute_on;
        options.robust = false;
        %options.stepsize_ratio = 1;
      

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        %[w_svrmu_nmf, infos_svrmu_nmf] = svrmu_nmf_old(V, K, options);
        [w_svrmu_nmf, infos_svrmu_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU';   
        sols{alg_idx} = w_svrmu_nmf;
        infos{alg_idx} = infos_svrmu_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_nmf.W, w_svrmu_nmf.H, zeros(F, N)) * 2 / dim;        
    end

    if svrmu_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on;
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-ACC';   
        sols{alg_idx} = w_svrmu_acc_nmf;
        infos{alg_idx} = infos_svrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_nmf.W, w_svrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;
    end
    
    if svrmu_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 0;
        options.ls = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on;
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-LS';   
        sols{alg_idx} = w_svrmu_acc_nmf;
        infos{alg_idx} = infos_svrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_nmf.W, w_svrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;
    end
        
    if svrmu_ls_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 1;
        options.ls = 1;
        options.rep_mode = 'fix';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on;
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-LS-ACC';   
        sols{alg_idx} = w_svrmu_acc_nmf;
        infos{alg_idx} = infos_svrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_nmf.W, w_svrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;        
    end   
    
    if svrmu_precon_ls_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 0;
        options.ls = 1;
        options.precon = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on;
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-Precon-LS';   
        sols{alg_idx} = w_svrmu_acc_nmf;
        infos{alg_idx} = infos_svrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_nmf.W, w_svrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;
    end    
    
    if svrmu_precon_ls_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 1;
        options.ls = 1;
        options.precon = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on;
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_nmf, infos_svrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-Precon-LS-ACC';   
        sols{alg_idx} = w_svrmu_acc_nmf;
        infos{alg_idx} = infos_svrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_nmf.W, w_svrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;
    end    

    if svrmu_acc_adaptive_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.permute_on = permute_on;
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 1;
        %options.rep_mode = 'fix';
        options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat;
        options.ratio = 1;  
        options.ratio = 0.3; % 1: original 1<: adaptive
        options.permute_on = permute_on;  
        options.robust = false;        

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_svrmu_acc_adaptive_nmf, infos_svrmu_acc_adaptive_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'SVRMU-ACC-Adaptive';   
        sols{alg_idx} = w_svrmu_acc_adaptive_nmf;
        infos{alg_idx} = infos_svrmu_acc_adaptive_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_svrmu_acc_adaptive_nmf.W, w_svrmu_acc_adaptive_nmf.H, zeros(F, N)) * 2 / dim;
    end


    if rsvrmu_acc_nmf_flag
        alg_idx = alg_idx + 1;  
        clear options;
        options.lambda = lambda;

        options.batch_size = batch_size;
        options.x_init = x_init;
        options.verbose = verbose;  
        options.permute_on = permute_on;
        options.fast_calc = 0;
        options.f_opt = f_opt;  
        options.repeat_inneriter = svrmu_inneriter;
        options.accel = 0;
        options.ls = 1;
        options.rep_mode = 'fix';
        %options.rep_mode = 'adaptive';
        options.w_repeat = 1;
        options.h_repeat = max_h_repeat; 
        options.ratio = 1;
        options.permute_on = permute_on; 
        options.robust = true; 

        options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1));    

        [w_rsvrmu_acc_nmf, infos_rsvrmu_acc_nmf] = svrmu_nmf(V, K, options);

        names{alg_idx} = 'R-SVRMU-LS';   
        sols{alg_idx} = w_rsvrmu_acc_nmf;
        infos{alg_idx} = infos_rsvrmu_acc_nmf;     
        costs{alg_idx} = nmf_cost(Vo, w_rsvrmu_acc_nmf.W, w_rsvrmu_acc_nmf.H, zeros(F, N)) * 2 / dim;
    end    

    alg_total = alg_idx;
    

    %% plot
    if plot_flag  
        %display_graph_icassp('epoch','cost', names, sols, infos);
        %display_graph_icassp('grad_calc_count','cost', names, sols, infos);
        %display_graph_icassp('time','cost', names, sols, infos);

        display_graph('numofgrad','cost', names, sols, infos);
        display_graph('time','cost', names, sols, infos);
        if f_opt ~= -Inf            
            display_graph('numofgrad','optimality_gap', names, sols, infos);
            display_graph('time','optimality_gap', names, sols, infos);
        end
    end
    

    
    % display original dic
    if oriimg_display_flag && ~isempty(classes)
        for ii = 1 : K
            k_index = find(classes==ii);
            col_img = Vo(:,k_index(1));
            col_img_reshape = reshape(col_img, img_in_dim);
            col_img_resize = imresize(col_img_reshape, img_out_dim);
            img(:,ii) = col_img_resize(:);
        end
        plot_dictionnary(img, [], dic_display_dim);    
    end

    clear results;

    for alg_idx = 1 : alg_total

        mse = calc_mse(Vo, sols{alg_idx}.W, sols{alg_idx}.H); 
        psnr = calc_psnr(Vo, sols{alg_idx}.W, sols{alg_idx}.H);
        fprintf('# %s:\t MSE:%e,\t PSNR:%e,\t time:%e [sec]\n', names{alg_idx}, mse, psnr, infos{alg_idx}.time(end));
       
        if clustering_flag
            cluster_num = length(unique(classes));
            Coeff = sols{alg_idx}.H;
            M = Coeff';        
            num_samples = size(M,1);
            perm_idx = randperm(num_samples);
            center = zeros(cluster_num, K);
            for i = 1 : cluster_num
                center(i,:) = M(perm_idx(i),:);
            end 
            
            % do k-menas        
            k_means_result = k_means(M, center, cluster_num);

            purity = calc_purity(classes, k_means_result);
            nmi = calc_nmi(classes, k_means_result);

            fprintf('# %s:\t Purity:%e,\t NMI: %e\n', names{alg_idx}, purity, nmi);        
        end

    end
    
    
    if dic_display_flag
        for alg_idx = 1 : alg_total
            clear img;
            for ii = 1 : K
                col_img = sols{alg_idx}.W(:,ii);
                col_img_reshape = reshape(col_img, img_in_dim);
                col_img_resize = imresize(col_img_reshape, img_out_dim);
                img(:,ii) = col_img_resize(:);
            end

            plot_dictionnary(img, [], dic_display_dim); 
            
        end
    end
    

    if denoise_display_flag && ~isempty(classes)
        for alg_idx = 1 : alg_total

            Rec = sols{alg_idx}.W * sols{alg_idx}.H; 

            for ii = 1 : K
                k_index = find(classes==ii);
                col_img = Rec(:,k_index(1));
                col_img_reshape = reshape(col_img, img_in_dim);
                col_img_resize = imresize(col_img_reshape, img_out_dim);
                img(:,ii) = col_img_resize(:);
            end

            plot_dictionnary(img, [], dic_display_dim); 

        end  
    end

end