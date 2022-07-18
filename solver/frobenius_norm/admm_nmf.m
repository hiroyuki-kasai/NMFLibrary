function [x, infos] = admm_nmf(V, rank, in_options)
% The alternating direction method of multipliers (ADMM) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} > 0.
%
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x n)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
% References
%
%
% This file is part of NMFLibrary
%
% This file has been ported from 
%       nnls_ADMM.m and FroNMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Created by H.Kasai on June 21, 2022.
%
% Change log: 
%
%       Jun. 24, 2022 (Hiroyuki Kasai): Added momentum acceleration mode
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.sub_mode = 'std';
    local_options.delta = 1e-6;
    local_options.inner_max_epoch = 500; 
    local_options.beta0 = 0.5;
    local_options.eta = 1.5; 
    local_options.gammabeta = 1.01;
    local_options.gammabetabar = 1.005; 
    local_options.momentum_h = 0; 
    local_options.momentum_w = 0; 
    local_options.scaling = true;
    local_options.warm_restart = false;    
    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    
    % initialize
    method_name = sprintf('ADMM (%s)', options.sub_mode);    
    epoch = 0;    
    grad_calc_count = 0; 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
    % intialize for ADMM
    if options.scaling
        [W, H] = normalize_WH(V, W, H, rank, 'type1');
    end
    
    [options, beta, betamax] = check_momemtum_setting(options);    
    
    if options.warm_restart
        nV = norm(V, 'fro');
        rel_error = zeros(1, options.max_epoch);
        rel_error(1) = sqrt(nV^2 - 2*sum(sum(V * H' .* W)) + sum(sum( H * H' .* (W'*W)))) / nV;          
    end
    W_prev = W; 
    H_prev = H; 
    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);


    if options.verbose > 1
        fprintf('ADMM (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.sub_mode, f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end
        
        %% update H        
        WtW = W' * W;
        WtX = W' * V;

        if ~isfield(options, 'rho')
            rho = trace(WtW) / rank; 
        else 
            rho = options.rho; 
        end
        
        inv_WtW_rho_I = inv(WtW+ rho*eye(rank)); 
        inv_WtW_rho_I_WtX = inv_WtW_rho_I * WtX; 
        
        % initialize Y and Z to zero 
        Y = zeros(rank, n); 
        Z = zeros(rank, n);
        cnt = 1; 
        Hp = Inf;         
        while (norm(H-Y,'fro') > options.delta*norm(H,'fro') || norm(H-Hp, 'fro') > options.delta*norm(H, 'fro'))... 
                    && cnt <= options.inner_max_epoch 
            Hp = H; 
            Y = max(0, H + Z/rho); 
            H = inv_WtW_rho_I_WtX + inv_WtW_rho_I * (rho*Y - Z); 
            Z = Z + rho*(H - Y); 
            cnt = cnt + 1;
        end
        H = max(H, 0);
        
        % perform momentum for H 
        if strcmp(options.sub_mode, 'momentum')
            [H, H_tmp1, H_tmp2] = do_momentum_h(H, H_prev, beta, epoch, options);
        end
                
        

        %% update W        
        HHt = H * H';
        HVt = H * V';
        
        % initialize for ADMM
        if ~isfield(options, 'rho')
            rho = trace(HHt) / rank; 
        else 
            rho = options.rho; 
        end           

        invHHtrhoI = inv(HHt+ rho*eye(rank)); 
        invHHtrhoIHVt = invHHtrhoI * HVt; 
        
        % initialize Y and Z to zero 
        Y = zeros(rank, m); 
        Z = zeros(rank, m);
        cnt = 1; 
        Wp = Inf; 
        Wt = W';      
        while (norm(Wt-Y, 'fro') > options.delta*norm(Wt, 'fro') || norm(Wt-Wp,'fro') > options.delta*norm(Wt, 'fro'))... 
                    && cnt <= options.inner_max_epoch 
            Wp = Wt; 
            Y = max(0, Wt + Z/rho); 
            Wt = invHHtrhoIHVt + invHHtrhoI * (rho * Y - Z); 
            Z = Z + rho*(Wt - Y); 
            cnt = cnt + 1;
        end
        Wt = max(Wt, 0); 
        W = Wt';
        
        % perform momentum for W 
        if strcmp(options.sub_mode, 'momentum')
            [W, H, W_tmp1] = do_momentum_w(W, W_prev, H, H_prev, H_tmp1, beta, epoch, options);
        end
        
        
        % perform warm_restart
        if options.warm_restart
            [W, H, W_prev, H_prev, rel_error, beta, betamax, options] = ...
                warm_restart(V, W, H, rank, W_prev, H_prev, W_tmp1, H_tmp1, H_tmp2, rel_error, beta, betamax, epoch, options);
        end        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);  
           
        % display info
        display_info(method_name, epoch, infos, options);
             
    end
    
    x.W = W;
    x.H = H;
    
end