function [x, infos] = mu_conv_nmf(V, rank, t, in_options)
% Multiplicative update (MU) based convolutive non-negative matrix factorization (MU-Conv-NMF).
%
% The problem of interest is defined as
%
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       matrix      V
%       rank        rank
%           
% Output:
%       w           solution of w
%       infos       information
%
% References:
%
%    
% This file is part of NMFLibrary
%
% This file has been ported by H.Kasai from 
% convNMF_MM1.m and convNMF_MM2.m at https://github.com/lyn202206/ADMM-Convolutive-NMF.
%
% Ported by H.Kasai on June 29, 2022
%
% Change log: 
%
%

    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.metric_type = 'beta-div';
    local_options.d_beta = 2;  
    local_options.alg = 'type1';
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);

    % initialize factors
    init_options = options;
    if ~isfield(options, 'x_init')
        W = zeros(m, rank, t);
        for i = 1 : t
            [init_factors, ~] = generate_init_factors(V, rank, init_options);    
            W(:, :, i) = init_factors.W;
        end
        H = init_factors.H;   
    else
        W = init_options.x_init.W;
        H = init_options.x_init.H;        
    end

    % initialize
    epoch = 0; 
    grad_calc_count = 0;

    options = check_divergence(options);
    sub_mode = sprintf('beta=%.1f', options.d_beta);
    if ~strcmp(options.metric_type, 'beta-div')
        sub_mode = options.metric_type;
    end
    method_name = sprintf('MU-Conv (%s)', sub_mode); 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm
    V_hat = zeros(m, n);
    for i = 0 : t-1
        V_hat = V_hat + W(:, :, i+1) * shift_t(H, i);
    end    

    % store initial info
    clear infos;   

    [Wcon, Hcon] = reconstruct_wh(W, H, t);
    [infos, f_val, optgap] = store_nmf_info(V, Wcon, Hcon, [], options, [], epoch, grad_calc_count, 0);
      
    
    if options.verbose > 1
        fprintf('MU-Conv (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', sub_mode, f_val, optgap); 
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
        
        % update H sequentially
        for i = 1 : n
            
            num = zeros(rank, 1); 
            denom = num;
            if i <= n-t+1
                for n_prime = i : i+t-1 
                    num = num + W(:, :, n_prime-i+1)' * ((V(:, n_prime) + eps) .* (V_hat(:, n_prime) + eps) .^ (options.d_beta - 2));
                    denom = denom + W(:, :, n_prime-i+1)'*(V_hat(:, n_prime)+eps).^(options.d_beta-1);
                end
            else
                for n_prime = i : n
                    num = num + W(:, :, n_prime-i+1)' * ((V(:, n_prime) + eps) .* (V_hat(:, n_prime) + eps) .^ (options.d_beta - 2));
                    denom = denom + W(:, :, n_prime-i+1)'*(V_hat(:, n_prime) + eps) .^ (options.d_beta-1);
                end
            end
            
            if strcmp(options.alg, 'type1')
                h_n_old = H(:, i);
                H(:, i) = H(:, i) .* (num./denom) .^ gamma_beta(options.d_beta);
                
                if i <= n-t+1
                    for n_prime = i:i+t-1
                        V_hat(:, n_prime) = max(V_hat(:, n_prime) + W(:, :, n_prime - i + 1) * (H(:, i) - h_n_old), 0);
                        % max(.,0) ensures the nonnegativity
                    end
                else
                    for n_prime = i:n
                        V_hat(:, n_prime) = max(V_hat(:, n_prime) + W(:, :, n_prime - i + 1) * (H(:, i) - h_n_old), 0);
                    end
                end
            else
                H(:, i) = H(:, i) .* (num./denom) .^ gamma_beta(options.d_beta);
            end
        
        end

        if strcmp(options.alg, 'type2')
            
            V_hat = zeros(m, n);
            for i = 0 : t-1
                V_hat = V_hat + W(:, :, i+1) * shift_t(H, i);
            end        
        end
        
        
        % update W
        for i = 0 : t - 1
            W_t_old = W(:, :, i+1);
            H_shift_t = shift_t(H, i);
            W(:, :, i+1) = W(:, :, i+1) .* (((((V + eps) .* (V_hat + eps) .^ (options.d_beta - 2)) * H_shift_t') ...
                ./ ((V_hat + eps).^(options.d_beta - 1) * H_shift_t'))) .^ gamma_beta(options.d_beta);
            V_hat = max(V_hat + (W(:, :, i+1) - W_t_old) * H_shift_t, 0); 
            % max(.,0) ensures nonnegativity
        end

        [W, H] = renormalize_convNMF(W,H);
        
        if strcmp(options.alg, 'type1')        
            % recalate the V_hat
            X_hat = zeros(size(V));
            for i=0:t-1
                tW = W(:, :, i+1);
                tH = shift_t(H, i);
                X_hat = X_hat + tW * tH;
            end
            V_hat = max(X_hat,0);
        end
        

        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        [Wcon, Hcon] = reconstruct_wh(W, H, t);        
        infos = store_nmf_info(V, Wcon, Hcon, [], options, infos, epoch, grad_calc_count, elapsed_time);          
       
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;   
    
end


function [W_concat, H_concat] = reconstruct_wh(W, H, t)
    
    W_concat = [];
    H_concat = [];  
    for j = 1 : t
        W_concat = [W_concat W(:, :,j)];
        H_concat = [H_concat; shift_t(H, j-1)]; 
    end 
end