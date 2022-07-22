function [x, infos] = kl_bmd_nmf(V, rank, in_options)
% Block mirror descent method for KL-based non-negative matrix factorization (KL-BMD-NMF).
%
% The problem of interest is defined as
%
%           min f(V, W, H),
%           where 
%           {V, W, H} >= 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
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
% This file has been ported from 
% BMD.m at https://github.com/LeThiKhanhHien/KLNMF written originally by LTK Hien.
%
% Ported by H.Kasai on June 28, 2022
%
% Change log: 
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.metric.type   = 'kl-div';
    local_options.myeps         = 1e-16;    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);        

    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    method_name = 'KL-BMD';    
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end       

    % initialize for this algorithm
    lambdaH = (1./(sum(V))); % the row which is the sum of columns of X
    lambdaH = repmat(lambdaH, rank, 1);
    lambdaW = (1./(sum(V,2)))';
    lambdaW = repmat(lambdaW, rank , 1);    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('KL-BMD: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
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

        % update H
        rj = sum(W)';
        rjc = repmat(rj, 1, n);
        bAv = V./(W*H+eps);
        cj = W' * bAv; 
        H = H ./ (1 + (lambdaH .* H) .* (rjc - cj));
        H = H + (H<options.myeps) .* options.myeps;        
       
        % update W
        rj = sum(H, 2);
        rjc = repmat(rj, 1, m);
        bAv = V' ./ (H' * W' + eps);
        cj = H * bAv; 
        Wt = W' ./ (1 + (lambdaW .* W') .* (rjc - cj));
        W = Wt';
        W = W + (W<options.myeps) .* options.myeps;         
        
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