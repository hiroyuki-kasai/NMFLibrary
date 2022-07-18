function [x, infos] = robust_online_mu_nmf(V, rank, in_options)
% Robust online non-negative matrix factorization (ONMF) with outliers (RONMF) algorithm.
%
% Inputs:
%       matrix      V
%       rank        rank
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       R. Zhao and Y. F. Tan,
%       "Online nonnegative matrix factorization with outliers,"
%       ICASSP, 2016.
%    
%
% This file is part of NMFLibrary.
%
% Created by H.Sakai and H.Kasai on Feb. 12, 2017
%
% Change log: 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];   
    local_options.lambda        = 1;
    local_options.x_init_robust = true;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    Wt = init_factors.W;
    H = init_factors.H; 
    R = init_factors.R; 
    
    % initialize
    method_name = 'Robust-Online-MU';
    epoch = 0;
    grad_calc_count = 0;
    l = zeros(m, options.batch_size) + options.lambda;    

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end    

    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end 
    
    % set start time
    start_time = tic();
    
    % main outer loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end  
        
        % Reset sufficient statistic        
        At = zeros(m, rank);
        Bt = zeros(rank, rank);        
        Ct = zeros(m, rank);

        % main inner loop
        for t = 1 : options.batch_size : n - 1

            % Retrieve vt, ht and rt
            vt = V(:, t:t+options.batch_size-1);
            ht = H(:, t:t+options.batch_size-1);
            rt = R(:, t:t+options.batch_size-1);

            % update ht/rt
            ht = ht .* (Wt.' * vt) ./ (Wt.' * (Wt * ht + rt));
            ht = ht + (ht<eps) .* eps;      
            rt = rt .* vt ./ (Wt * ht + rt + l);

            % update sufficient statistics
            At = At + vt *  ht';
            Bt = Bt + ht *  ht'; 
            Ct = Ct + rt *  ht';

            % Update Wt
            Wt = Wt .* At ./ (Wt * Bt + Ct); 
            Wt = Wt + (Wt<eps) .* eps;

            % Update H
            H(:,t:t+options.batch_size-1) = ht;    
            
            % Update R
            R(:,t:t+options.batch_size-1) = rt;
            
            grad_calc_count = grad_calc_count + m * options.batch_size;
        end
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        infos = store_nmf_info(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display info
        display_info(method, epoch, infos, options);
     
    end
    
    x.W = Wt;
    x.H = H;
    x.R = R; 

end