function [x, infos] = fro_mu_partial_nmf(V, rank, in_options)
%
% This file is part of NMFLibrary
%
% Created by H.Kasai on Feb. 16, 2017
%
% Change log: 
%
%       June 16, 2022 (Hiroyuki Kasai): Initial version.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.alg       = 'mu';
    local_options.norm_h    = 0;
    local_options.norm_w    = 1;    
    local_options.alpha     = 2;
    local_options.delta     = 0.1;
    local_options.myeps     = 1e-16;
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);        
    
    if ~strcmp(options.alg, 'mu')
        fprintf('Invalid algorithm: %s. Therfore, we use mu (i.e., multiplicative update).\n', options.alg);
        options.alg = 'mu';
    end
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    method_name = sprintf('MU-Partial (%s:%s)', options.alg, options.metric_type);
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('MU-Partial (%s:%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, options.metric, f_val, optgap); 
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

        if strcmp(options.alg, 'mu')
                
            % update H
            H = H .* (W' * V) ./ (W' * W * H);
            H = H + (H<options.myeps) .* options.myeps;

            % update W
            W(:, options.updateW) = W(:, options.updateW) .* (V * H(options.updateW, :)') ...                
                ./ (W * (H * H(options.updateW, :)'));                
            W = W + (W<options.myeps) .* options.myeps;

        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos = store_nmf_info(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;
    
end