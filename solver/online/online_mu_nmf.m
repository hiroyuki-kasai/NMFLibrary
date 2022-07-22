function [x, infos] = online_mu_nmf(V, rank, in_options)
% Online non-negative matrix factorization (Online-NMF) algorithm.
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
%       S. S. Bucak, B. Gunsel,
%       "Incremental Subspace Learning via Non-negative Matrix Factorization,"
%       Pattern Recognition, 2009.
%    
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai and H.Sakai on Feb. 12, 2017
%
% Change log: 
%
%       Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    
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
    Wt = init_factors.W;
    H = init_factors.H; 
    R = init_factors.R; 
    
    % initialize
    method_name = 'Online-MU';    
    epoch = 0;
    grad_calc_count = 0;

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

        % main inner loop
        for t = 1 : options.batch_size : n - 1

            % Retrieve vt and ht
            vt = V(:, t:t+options.batch_size-1);
            ht = H(:, t:t+options.batch_size-1);

            % uddate ht
            ht = ht .* (Wt.' * vt) ./ (Wt.' * (Wt * ht));
            ht = ht + (ht<eps) .* eps;      
            
            % update sufficient statistics
            At = At + vt * ht';
            Bt = Bt + ht * ht';              

            % update W
            Wt = Wt .* At ./ (Wt * Bt); 
            Wt = Wt + (Wt<eps) .* eps;

            % store new h
            H(:,t:t+options.batch_size-1) = ht;  
            
            grad_calc_count = grad_calc_count + m * options.batch_size;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        infos = store_nmf_info(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
                
        % display info
        display_info(method_name, epoch, infos, options);
        
    end
    
    x.W = Wt;
    x.H = H;

end