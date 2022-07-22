function [x, infos] = minvol_nmf(V, rank, in_options)
% Minimum-volume rank-deficient nonnegative matrix factorizations algorithm.
%
% The problem of interest is defined as
%
%       min ||M-WH||_F^2 + lambda' * logdet(W^TW + delta I) ,
%       where 
%       {W, H} >= 0.
%       and sum-to-one constraints on W or H: 
%       H^T e <= e (model 1), or 
%       H e    = e (model 2), or 
%       W^T e  = e (model 3, default). 
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       V. Leplat, A.M.S. Ang, N. Gillis, 
%       "Minimum-volume rank-deficient nonnegative matrix factorizations", 
%       ICASSP 2019, May 12-17, 2019, 
%
%
% This file is part of NMFLibrary
%
% This file has been ported from 
%   minvol_nmf.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%   by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June. 21, 2022 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.model  = 3;
    local_options.delta  = 0.1;
    local_options.lambda = 0.1;
    local_options.inner_max_epoch = 10;

    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H; 

    % initialize
    method_name = 'minvol-NMF';
    epoch = 0; 
    grad_calc_count = 0;
    
    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm
    if options.model == 1
        options.proj = 1;
        normalize_WH_type = 'type3';
    elseif options.model == 2
        options.proj = 2;
        normalize_WH_type = 'type4';        
    elseif options.model == 3
        options.proj = 0;
        normalize_WH_type = 'type5';        
    end

    %[W,H] = normalizeWH(W,H,options.model,V); 
    [W,H] = normalize_WH(V, W, H, rank, normalize_WH_type);
    normV2 = sum(V(:).^2);
    normV = sqrt(normV2); 
    WtW = W'*W;
    WtV = W'*V;
    err1 = max(0, normV2-2*sum(sum(WtV.*H)) + sum(sum(WtW.*(H*H'))));
    err2 = log(det(WtW+options.delta*eye(rank)));
    options.lambda = options.lambda * max(1e-6,err1) / (abs(err2));

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
        
        % update W
        VHt = V * H';
        HHt = H * H';
        Y = inv((W' * W + options.delta * eye(rank)));
        A = options.lambda * Y + HHt;
        if options.model <= 2
            W = FGMqpnonneg(A, VHt, W, options.inner_max_epoch, 1); 
        elseif options.model == 3
            W = FGMqpnonneg(A, VHt, W, options.inner_max_epoch, 2); 
        end

        % update H
        options_fpgm.init = H;
        options_fpgm.inner_max_epoch = options.inner_max_epoch;
        options_fpgm.delta = options.delta;
        [H,WtW,WtV] = nnls_fpgm(V, W, options_fpgm);         
        
        
        err1 = max(0, normV2 - 2*sum(sum(WtV.*H)) + sum(sum(WtW.*(H*H'))));

        % Tuning lambda to obtain options.target relative error 
        if isfield(options,'target')
            if sqrt(err1)/normV > options.target+0.001
                options.lambda = options.lambda*0.95;
            elseif sqrt(err1)/normV < options.target-0.001
                options.lambda = options.lambda*1.05;
            end
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