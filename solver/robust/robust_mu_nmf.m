function [x, infos] = robust_mu_nmf(V, rank, in_options)
% Robust non-negative matrix factorization (NMF) with outliers (RNMF) algorithm.
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
%       Naiyang Guan, Dacheng Tao, Zhigang Luo, and Bo Yuan,
%       "Online nonnegative matrix factorization with robust stochastic approximation,"
%       IEEE Trans. Newral Netw. Learn. Syst., 2012.
%    
%
% This file is part of NMFLibrary.
%
% Created by H.Sakai and H.Kasai on Feb. 12, 2017
%
% Change log: 
%
%       May. 21, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.lambda        = 1;
    local_options.x_init_robust = true;
    
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
    R = init_factors.R; 

    % initialize
    method_name = 'Robust-MU';        
    epoch = 0; 
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm
    L = zeros(m, n) + options.lambda;     
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('Robust-MU: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
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
        
        % update H/R/W
        H = H .* (W.' * V) ./ (W.' * (W * H + R));
        R = R .* V./ (W * H + R + L);
        W = W .* (V * H.') ./ ((W * H + R) * H.');
        
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        infos = store_nmf_info(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;
    x.R = R; 

end