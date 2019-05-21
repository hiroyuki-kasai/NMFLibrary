function [x, infos] = rnmf(V, rank, in_options)
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
% Created by H.Sakai and H.Kasai on Feb. 12, 2017
%
% Change log: 
%
%   May. 21, 2019 (Hiroyuki Kasai): Added initialization module.
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
        fprintf('# R-NMF: started ...\n');           
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H; 
    R = init_factors.R; 

    % initialize
    epoch = 0; 
    L = zeros(m, n) + options.lambda; 
    grad_calc_count = 0;
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);        
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('R-NMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();
    prev_time = start_time;

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) 
        
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
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('R-NMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
       
        prev_time = elapsed_time;
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# R-NMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# R-NMF: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;
    x.R = R;      
end


