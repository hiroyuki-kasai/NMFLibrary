function [x, infos] = palm_sparse_smooth_nmf(V, rank, in_options)
% PALM framework with smoothness and sparsity constraints for non-negative
% matrix factorization (PALM-Sparse-Smooth-NMF)
%
% The problem of interest is defined as
%
%           min || WH - V ||_F^2 + lambda * || W ||_1 + eta || HT ||_F^2 
%                                   + betaW ||W||_F^2 + betaH ||H||_F^2,
%           where 
%           {V, W, H} >= 0, 
%
%           Algorithm for NMF with eucidian norm as objective function and 
%           L1 constraint on W for sparse paterns and Tikhonov regularization 
%           for smooth activation coefficients.
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
%           lambda      weight for the L1 sparsity penalty (default: 0)
%           eta         weight for the smoothness constraint.
%           gamma1: 	constant > 1 for the gradient descend step of W.
%           gamma2:     constant > 1 for the gradient descend step of W.
%           betaH:      constant. L-2 constraint for H.
%           betaW:      constant. L-2 constraint for W.
% Output:
%       w           solution of w
%       infos       information
%
% References:
%    
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%       palm_nmf.m at https://github.com/raimon-fa/palm-nmf
%
%{
The MIT License
Copyright (c) 2017 Raimon Fabregat
Permission is hereby granted, free of charge, 
to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to 
deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom 
the Software is furnished to do so, 
subject to the following conditions:
The above copyright notice and this permission notice 
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%}
%
% Ported by M.Horie and H.Kasai on June 21, 2022 for NMFLibrary
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.lambda = 0;   % sparsity = lambda
    local_options.eta   = 0;    % smoothness = eta
    local_options.betaW = 0.1;
    local_options.betaH = 0.1;
    local_options.gamma1 = 1.001;
    local_options.gamma2 = 1.001;
    local_options.sub_mode = 'std';
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    if options.verbose > 0
        fprintf('# PALM-Sparse-Smooth-NMF: started ...\n');           
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H; 

    % initialize for PAML-NMF
    TTp_norm = 0;
    TTp = zeros(n);
    if options.lambda == 0 && options.eta == 0
        % NMF
        % In the case without constraints it can be shown that 
        % the gammas can be divided by 2 (Bolte 2014)
        options.gamma1 = options.gamma1 / 2;
        options.gamma2 = options.gamma2 / 2;
        options.betaW = 0;    
        options.betaH = 0;          
    elseif options.lambda > 0 && options.eta == 0
        % sparse NMF       
        options.betaW = 0;
        options.sub_mode = 'sparse';        
    elseif options.lambda == 0 && options.eta > 0
        % smooth NMF
        % Tikhonov regularization matrix
        T = eye(n) - diag(ones(n-1, 1),-1);
        T = T(:, 1:end-1);
        TTp = T * T';
        TTp_norm = norm(TTp, 'fro');
        options.betaH = 0;  
        options.sub_mode = 'smooth';          
    elseif options.lambda > 0 && options.eta > 0
        % smooth and sparse NMF
        % Tikhonov regularization matrix
        T = eye(n) - diag(ones(n-1, 1),-1);
        T = T(:, 1:end-1);
        TTp = T * T';
        TTp_norm = norm(TTp, 'fro');
        options.sub_mode = 'smooth & sparse';         
    else
        error('Give positive values to the parameters')
    end
    
    % initialize
    method_name = sprintf('PALM-Sparse-Smooth (%s)', options.sub_mode); 
    epoch = 0; 
    grad_calc_count = 0;

         
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, options.sub_mode, f_val, optgap); 
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

        % update W
        c = options.gamma1 * 2 * (norm(H * H', 'fro') + options.betaW);    
        z1 = W - (1 / c) * 2 * ((W * H - V) * H' + options.betaW * W);
        W = max(z1 - 2 * options.lambda / c, 0);
        
        % update H        
        d = options.gamma2 * 2 * (norm(W*W', 'fro') + options.eta * TTp_norm + options.betaH);
        z2 = H - (1 / d) * 2 * (W' * (W * H - V) + options.eta * (H * TTp) + options.betaH * H);   
        H = max(z2, 0);

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