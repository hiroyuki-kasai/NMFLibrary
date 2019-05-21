function [x, infos] = onmf(V, rank, in_options)
% Online non-negative matrix factorization (ONMF) algorithm.
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
% Created by H.Kasai and H.Sakai on Feb. 12, 2017
%
% Change log: 
%
%   Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    if options.verbose > 0
        fprintf('# ONMF: started ...\n');           
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    Wt = init_factors.W;
    H = init_factors.H; 
    R = init_factors.R; 
    
    % initialize
    epoch = 0;
    grad_calc_count = 0;
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('ONMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end    
    
    % set start time
    start_time = tic();
    prev_time = start_time;
    
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)        
        
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
        [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('ONMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
        
        prev_time = elapsed_time;             
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# ONMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# ONMF: Max epoch reached (%g).\n', options.max_epoch);
        end     
    end
    
    x.W = Wt;
    x.H = H;
end





