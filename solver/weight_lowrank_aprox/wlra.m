function [x, infos] = wlra(V, rank, P, in_options)
% Weighted Low-Rank matrix Approximation algorithm.
%
% The problem of interest is defined as
%
%       min_{W, H}  ||V-WH^T||_P^2 + lambda (||W||_F^2+||H||_F^2),
%       where 
%       ||V-WH^T||_P^2 = sum_{i,j} P(i,j) (V-WH^T)_{i,j}^2.
%
% It is possible to requires (W,H) >= 0 using options.nonneg = 1. 
%
% Inputs:
%       matrix      V
%       rank        rank
%       P           (m x n) nonnegative weight matrix
%       options     options
%           nonneg  1: W>=0 and H>=0 (default=0)
%           lambda  penalization parameter (default=1e-6)
%           
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       No available
%
%
% This file is part of NMFLibrary.
%
%       This file has been ported from 
%       WLRA.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 08, 2022 (Mitsuhiko Horie): Ported initial version 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.nonneg = 0;                   % if nonneg = 1: W>=0 and H>=0 (default=0)
    local_options.lambda = 1e-6;
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end     
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H; 

    % initialize
    method_name = 'WLRA';
    epoch = 0; 
    grad_calc_count = 0;
    lambda = options.lambda;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end       
    
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
        
        R = V - W * H; 
        
        for k = 1 : rank
            R = R + W(:, k)*H(k, :); 
            Rp = R .* P;
            W(:, k) = (Rp * H(k, :)') ./ (P * (H(k, :)'.^2) + lambda);
            
            if options.nonneg == 1
                W(:, k) = max(eps, W(:, k)); 
            end
            
            H(k,:) = ((Rp' * W(:, k)) ./ (P' * (W(:, k).^2) + lambda))';
            
            if options.nonneg == 1
                H(k, :) = max(eps, H(k, :)); 
            end
            R = R - W(:, k) * H(k, :);
        end
        
        [W, H] = scalingWH(W, H); 
        
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

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

% Scaling of columns of W (mxr) and H (rxn) to have 
% ||W(:,k)|| = ||H(k,:)|| for all k
function [W, H] = scalingWH(W, H)
    [m, r] = size(W); 
    normW = sqrt((sum(W.^2))) + 1e-16;
    normH = sqrt((sum(H'.^2))) + 1e-16;
    for k = 1 : r
        W(:, k) = W(:, k) / sqrt(normW(k)) * sqrt(normH(k));
        H(k, :) = H(k, :) / sqrt(normH(k)) * sqrt(normW(k));
    end
end