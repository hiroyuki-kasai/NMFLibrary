function [x, infos] = prob_nmf(V, rank, in_options)
% Probabilistic Non-negative Matrix Factorization (Prob-NMF)
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} >= 0.
%%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options    
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x n)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
%
% Reference:
%
%
% This file is part of NMFLibrary.
%
%       This file has been ported from NMF DTU Toolbox.
%       Lars Kai Hansen, IMM-DTU (c) November 2005
%
% Change log: 
%
%       June 17, 2022 (Hiroyuki Kasai): Ported initial version 
%

    
    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];

    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 

    % initialize factors
    init_options = options;
    init_options.norm_w = true;
    init_options.norm_h = true;    
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;     
    
    % initialize
    method_name = 'Prob-NM';
    epoch = 0;    
    grad_calc_count = 0; 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     

    % initialize for this algorithm
    V_factor = (sum(sum(V)));
    V = V / V_factor;
    P = ones(rank,1);
    P = P/sum(P);
    W1 = W;
    H1 = H;
     
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

        % E-step
        Qnorm = (W * diag(P)) * H;

        for k=1:rank

            % E-step
            Q = (W(:, k) * H(k, :) * P(k)) ./ (Qnorm+eps);
            VQ = V .* Q;

            % M-step W
            dummy = sum(VQ,2);
            W1(:, k) = dummy / (sum(dummy));
            dummy = sum(VQ,1);
            H1(k, :) = dummy / (sum(dummy));
        end
    
        W = W1;
        H = H1;   
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
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