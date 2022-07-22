function [x, infos] = sparse_nmf(V, rank, in_options)
% Sparse nonnegative matrix factorization (sparseNMF)
%
% The problem of interest is defined as
%
%       min D(V||W*H) + lambda * sum(H(:)),
%       where 
%       {V, W, H} > 0.
%
%       L1-based sparsity constraint on H.
%       Normalizes W column-wise.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
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
% Reference:
%
%
% This file is part of NMFLibrary.
%
% Created by Patrik Hoyer, 2006 (and modified by Silja Polvi-Huttunen, 
% University of Helsinki, Finland, 2014)
%
% Modified by H.Kasai on Jul. 23, 2018
%
% Change log: 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jul. 14, 2022 (Hiroyuki Kasai): Fixed algorithm.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options 
    local_options.lambda = 0;   % regularizer for sparsity
    local_options.cost  = 'euc'; % 'euc' or 'kl-div'
    
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
    
    % initialize
    method_name = 'sparseNMF';
    epoch = 0;    
    grad_calc_count = 0; 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    % store additionally different cost
    reg_val = options.lambda*sum(sum(H));
    f_val_total = f_val + reg_val;
    infos.cost_reg = reg_val;
    infos.cost_total = f_val_total;       
    
    if options.verbose > 1
        fprintf('sparseNMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
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

        % update H with MU
        %H = (H.*(W'*(V./(W*H))))/(1+alpha);
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        H = (H.*(W'*VC))/(1+options.lambda);
        
      
        % update W by Lee and Seung's divergence step
        %W = W.*((V./(W*H))*H')./(ones(vdim,1)*sum(H'));
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        W = W.*(VC*H')./(ones(m,1)*sum(H,2)');     
        
        % Liu, Zheng, and Lu add this normalization step
        W = W./(ones(m,1)*sum(W));        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        % store additionally different cost
        reg_val = options.lambda*sum(sum(H));
        f_val_total = f_val + reg_val;
        infos.cost_reg = [infos.cost_reg reg_val];
        infos.cost_total = [infos.cost_total f_val_total];   

        % display info
        display_info(method_name, epoch, infos, options);            

    end 

    x.W = W;
    x.H = H;
    
end