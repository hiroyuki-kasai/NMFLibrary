function [x, infos] = projective_nmf(V, rank, in_options)
% Projective non-negative matrix factorization (projectiveNMF).
%
% The problem of interest is defined as
%
%       min ||V - WW^TV||_F^2,
%       where 
%       W >= 0.
%
% Given a non-negative matrix V, factorized non-negative matrices W is calculated.
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
%       Z. Yang, and E. Oja,
%       "Linear and nonlinear projective nonnegative matrix factorization,"
%       IEEE Transactions on Neural Networks, 21(5), pp.734-749, 2010.
%  
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%       projectiveNMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Ported by T.Fukunaga and H.Kasai on June 24, 2022 for NMFLibrary
%
% Change log: 
%
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.delta = 1e-4;
    local_options.special_stop_condition = @(epoch, infos, options, stop_options) projective_nmf_stop_func(epoch, infos, options, stop_options);
    
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

    % initialize
    method_name = 'Projective-NMF';
    epoch = 0; 
    grad_calc_count = 0;
    stop_options = [];

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
    % initialize for this algorithm
    Wp = zeros(size(W)); 
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W*W', V, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();

    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch) && (norm(W - Wp, 'fro') > options.delta*norm(W, 'fro'))
    while true
        
        % check stop condition
        stop_options.W = W;
        stop_options.Wp = Wp;        
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options, stop_options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end        
        
        % update 
        Wp = W; 
        VtW = V' * W;   % n by r 
        VVtW = V * VtW; % m by r 
        WtW = W' * W;   % r by r
        VtWtVtW = VtW' * VtW; % r by r 
        
        % Optimal scale the initial solution
        % This is important otherwise the algorithm oscillates
        alpha = sum(sum(VVtW .* W)) / sum(sum(WtW .* VtWtVtW));
        W = W * sqrt(alpha);
        
        % update the other factors accordingly
        VVtW = sqrt(alpha) * VVtW;
        WtW = alpha * WtW;
        VtWtVtW = alpha * VtWtVtW;
        
        % multiplicative update by Yang and Oja
        W = W .* (2 * VVtW) ./ (W * (VtWtVtW) + VVtW * (WtW)); 
      
        % measure elapsed time
        elapsed_time = toc(start_time);   
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;           

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        infos = store_nmf_info(V, W*W', V, [], options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W; 

end


function [stop_flag, reason, rev_infos] = projective_nmf_stop_func(epoch, infos, options, stop_options)

    stop_flag = false;
    reason = [];
    rev_infos = [];

    W = stop_options.W;
    Wp = stop_options.Wp;

    if (norm(W - Wp, 'fro') < options.delta * norm(W, 'fro'))
        stop_flag = true;
        reason = sprintf('Solution change tolerance reached: norm(W-Wp) = %.4e < options.delta * norm(W) %.4e)\n', norm(W-Wp,'fro'), options.delta * norm(W,'fro'));
    end

end