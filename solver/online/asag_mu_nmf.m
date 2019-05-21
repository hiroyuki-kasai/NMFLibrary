function [x, infos] = asag_mu_nmf(V, rank, in_options)
% multiplicative update for non-negative matrix factorization (SAGA-MU-NMF) algorithm.
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
%       Romain Serizel, Slim Essid and Ga?l Richard,
%       "Mini-batch stochastic approaches for accelerated multiplicative updates 
%       in nonnegative matrix factorisation with beta-divergence,"
%       IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP), 
%       MLSP2016.
%
%    
% Created by H.Kasai on March. 22, 2017
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
    local_options.lambda = 1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    if options.verbose > 0
        fprintf('# ASAG-MU-NM: started ...\n');           
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
    
    % permute samples
    if options.permute_on
        perm_idx = randperm(n);
    else
        perm_idx = 1:n;
    end  
    
    V = V(:,perm_idx);
    H = H(:,perm_idx);   
    
    % prepare Delta_minus and Delta_plus
    Delta_minus = zeros(m, rank);
    Delta_plus = zeros(m, rank); 
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);     
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('ASAG-MU-NMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();
    prev_time = start_time;    
    
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           
        
        cnt = 0;
        % main inner loop
        for t = 1 : options.batch_size : n - 1
            cnt = cnt + 1;

            % retrieve vt and ht
            vt = V(:,t:t+options.batch_size-1);
            ht = H(:,t:t+options.batch_size-1);
            
            % uddate ht
            ht = ht .* (Wt.' * vt) ./ (Wt.' * (Wt * ht));
            ht = ht + (ht<eps) .* eps; 

            % update Delta_minus and Delta_plus
            Delta_minus = (1-options.lambda) * Delta_minus + options.lambda * vt * ht';
            Delta_plus = (1-options.lambda) * Delta_plus + options.lambda * Wt * (ht * ht');

            % update W
            Wt = Wt .* (Delta_minus ./Delta_plus);            
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
            if ~mod(epoch,disp_freq)
                fprintf('ASAG-MU-NMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
        
        prev_time = elapsed_time;          
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# ASAG-MU-NMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# ASAG-MU-NMF: Max epoch reached (%g).\n', options.max_epoch);
        end     
    end
    
    x.W = Wt;
    x.H(:,perm_idx) = H;
end





