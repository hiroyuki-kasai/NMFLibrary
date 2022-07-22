function [x, infos] = alternating_onmf(V, rank, in_options)
% Orthogonal two-block coordinate descent (2-BCD) for non-negative matrix factorization (Alt-Orth-NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} >= 0, and W or H is orthogonal, i.e., HH^T = I_r. 
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
%       F. Pompili, N. Gillis, P.-A. Absil and F. Glineur, 
%       "Two Algorithms for Orthogonal Nonnegative Matrix Factorization
%       with Application to Clustering," 
%       Neurocomputing 141, pp. 15-25, 2014. 
%    
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
% alternatingONMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
% by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 14, 2022 (Hiroyuki Kasai): Ported initial version 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.orth_h = true;
    local_options.delta = 0;
    local_options.special_stop_condition = @(epoch, infos, options, stop_options) alt_nmf_stop_func(epoch, infos, options, stop_options);    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W; % use only W
    
    % initialize
    method_name = 'ALT-ONMF';       
    epoch = 0; 
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm    
    % Xn: normalized version of X, ||Xn(:,j)||_2  1 for all j
    norm2x = sqrt(sum(V.^2,1)); 
    Vn = V .* repmat(1./(norm2x+1e-16), m, 1);  
    normV2 = sum(V(:).^2); 
    stop_options.normV2 = normV2;
    H = nnls_orth(V, W, Vn);
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    orth_val = norm(H*H' - eye(rank), 'fro');
    [infos.orth] = orth_val;
    infos.prev_error =  Inf;  
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();

    % main loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag, infos] = check_stop_condition(epoch, infos, options, stop_options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end
        
        % update H
        % H = argmin_H ||X-WH||_F, H >= 0, rows H orthogonal up to a scaling of the rows of H
        H = nnls_orth(V, W, Vn); 

        % normalize rows of H 
        norm2h = sqrt(sum(H'.^2, 1)) + 1e-16;
        H = repmat(1./norm2h', 1, n) .* H;   

        % update W
        W = V * H';
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);          
        orth_val = norm(H*H' - eye(rank), 'fro');
        [infos.orth] = [infos.orth orth_val];
     
        % display info
        display_info(method_name, epoch, infos, options);

        % check convergence (by original code)
        e = sqrt( (normV2-sum(sum(W.^2)))/normV2 ); 
        if (epoch > 2) && (abs(e_prev-e) < options.delta)
            %break;
        else
            e_prev = e;
        end
    end
    
    x.W = W;
    x.H = H;

end


function [stop_flag, reason, infos] = alt_nmf_stop_func(epoch, infos, options, stop_options)

    stop_flag = false;
    reason = [];

    normV2 = stop_options.normV2;
    W = infos.final_W;
    
    error = sqrt( (normV2-sum(sum(W.^2)))/normV2 ); 
    if (epoch > 2) && (abs(infos.prev_error - error) < options.delta)
        stop_flag = true;
        reason = sprintf('Relative solution change tolerance reached: %.4e < %.4e\n', abs(infos.prev_error - error), options.delta);
    else
        infos.prev_error = error;
    end

end