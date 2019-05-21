function [x, infos] = ronmf(V, rank, in_options)
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
% Created by H.Sakai and H.Kasai on Feb. 12, 2017
%
% Change log: 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
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
    
    if options.verbose > 0
        fprintf('# R-ONMF: started ...\n');           
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    Wt = init_factors.W;
    H = init_factors.H; 
    R = init_factors.R; 
    
    % initialize
    epoch = 0;
    l = zeros(m, options.batch_size) + options.lambda;
    grad_calc_count = 0;
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('R-ONMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end 
    
    % set start time
    start_time = tic();
    prev_time = start_time;        
    
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)   
        
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
        [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('R-ONMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
        
        prev_time = elapsed_time;          
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# R-ONMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# R-ONMF: Max epoch reached (%g).\n', options.max_epoch);
        end     
    end 
    
    x.W = Wt;
    x.H = H;
    x.R = R;    
end





