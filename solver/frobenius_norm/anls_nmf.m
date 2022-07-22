function [x, infos] = anls_nmf(V, rank, in_options)
% Alternative non-negative least squares (ANLS) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options
%
%
% References:
%       Jingu Kim, Yunlong He, and Haesun Park,
%       "Algorithms for Nonnegative Matrix and Tensor Factorizations: A Unified View 
%       Based on Block Coordinate Descent Framework,"
%       Journal of Global Optimization, 58(2), pp. 285-319, 2014.
%
%       Jingu Kim and Haesun Park.
%       "Fast Nonnegative Matrix Factorization: An Active-set-like Method and Comparisons,"
%       SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
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
%
% This file is part of NMFLibrary
%
% This code calls functions {nnls1_asgivens, nnlsm_activeset, nnlsm_blockpivot}
% written by Jingu Kim. See https://github.com/kimjingu/nonnegfac-matlab.
%
% Ported by H.Kasai on Apr. 04, 2017
%
% Change log: 
%
%       Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jun. 24, 2022 (Hiroyuki Kasai): Added momentum acceleration mode and mofified.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.alg   = 'anls_asgroup';
    local_options.sub_mode = 'std';    
    local_options.delta = 0.1;
    local_options.inner_max_epoch = 500;
    local_options.inner_max_epoch_parameter = 0.5;       
    local_options.beta0 = 0.5;
    local_options.eta = 1.5; 
    local_options.gammabeta = 1.01;
    local_options.gammabetabar = 1.005; 
    local_options.momentum_h = 0; 
    local_options.momentum_w = 0; 
    local_options.scaling = true;
    local_options.warm_restart = false;   

    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);    

    % set paramters
    if ~strcmp(options.alg, 'anls_asgroup') && ~strcmp(options.alg, 'anls_asgivens') ...
            && ~strcmp(options.alg, 'anls_bpp') 
        fprintf('Invalid algorithm: %s. Therfore, we use anls_asgroup (i.e., ANLS with Active Set Method and Column Grouping).\n', options.alg);
        options.alg = 'anls_asgroup';
    end
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    method_name = sprintf('ANLS (%s)', options.alg);
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     
    
    if options.scaling
        [W, H] = normalize_WH(V, W, H, rank, 'type1');
    end
    
    % initialize for ANLS
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
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
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
        if strcmp(options.alg, 'anls_asgroup')
            ow = 0;
            
            H = nnlsm_activeset(W'*W, W'*V, ow, 1, H);

        elseif strcmp(options.alg, 'anls_asgivens')
            ow = 0;
            
            WtV = W' * V;
            for i=1:size(H,2)
                H(:,i) = nnls1_asgivens(W'*W, WtV(:,i), ow, 1, H(:,i));
            end

        elseif strcmp(options.alg, 'anls_bpp')
            
            H = nnlsm_blockpivot(W'*W, W'*V, 1, H);
            
        end
        
        % perform momentum for H
        if strcmp(options.sub_mode, 'momentum')
            [H, H_tmp1, H_tmp2] = do_momentum_h(H, H_prev, beta, epoch, options);
        end
        
        
        
        %% update W
        if strcmp(options.alg, 'anls_asgroup')
            ow = 0;
            
            W = nnlsm_activeset(H*H', H*V', ow, 1, W');
            W = W';
            
        elseif strcmp(options.alg, 'anls_asgivens')
            ow = 0;
            
            HAt = H * V';
            Wt = W';
            for i=1:size(W,1)
                Wt(:,i) = nnls1_asgivens(H*H', HAt(:,i), ow, 1, Wt(:,i));
            end
            W = Wt';
            
        elseif strcmp(options.alg, 'anls_bpp')
            
            W = nnlsm_blockpivot(H*H', H*V', 1, W');
            W = W';

        end 
        
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