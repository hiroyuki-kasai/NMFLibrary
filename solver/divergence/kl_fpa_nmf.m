function [x, infos] = kl_fpa_nmf(V, rank, in_options)
% First-order primal-dual algorithm (FPA) based on the Chambolle-Pock algorithm
% for non-negative matrix factorization (KL-FPA-NMF).
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
%       Felipe Yanez, and Francis Bach. 
%       "Primal-Dual Algorithms for Non-negative Matrix Factorization with the 
%       Kullback-Leibler Divergence,"
%       IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
%       2017.
%    
%
% This file is part of NMFLibrary
%
% This file has been ported from 
%   nmf_kl_fpa.m at https://github.com/felipeyanez/nmf
%   by Felipe Yanez
%
%   Copyright (c) 2014-2016 Felipe Yanez
%
%   Permission is hereby granted, free of charge, to any person obtaining a 
%   copy of this software and associated documentation files (the "Software"), 
%   to deal in the Software without restriction, including without limitation 
%   the rights to use, copy, modify, merge, publish, distribute, sublicense, 
%   and/or sell copies of the Software, and to permit persons to whom the 
%   Software is furnished to do so, subject to the following conditions:
%
%   The above copyright notice and this permission notice shall be included 
%   in all copies or substantial portions of the Software.
%
%   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
%   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
%   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
%   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
%   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
%   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
%   OTHER DEALINGS IN THE SOFTWARE.
%
%
% Ported by M.Horie and H.Kasai on June 21, 2022
%
% Change log: 
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];   
    local_options.metric_type = 'kl-div';    
    local_options.inner_max_epoch = 5;
    
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
    method_name = 'KL-FPA';
    epoch = 0; 
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm    
    chi   = -V ./ (W * H);
    % chi = bsxfun(@times, chi, 1./max(bsxfun(@times, -W'*chi, 1./sum(W,1)')));
    chi   =  chi .* (1 ./ max((-W' * chi) .* (1 ./ sum(W, 1)')));
    Wbar  = W;
    Wold  = W;
    Hbar  = H;
    Hold  = H;
    
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

        % update H
        sigma = sqrt(m / rank) * sum(W(:)) ./ sum(V,1)  / norm(W);
        tau   = sqrt(rank / m) * sum(V,1)  ./ sum(W(:)) / norm(W);

        for j = 1:options.inner_max_epoch
            % chi  = chi + bsxfun(@times, W*Hbar, sigma);
            % chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
            % H    = max(H - bsxfun(@times, W'*(chi + 1), tau), 0);
            chi  = chi + W * Hbar .* sigma;
            chi  = (chi - sqrt(chi.^2 + V .* (4*sigma))) / 2;
            H    = max(H - (W' * (chi + 1) .* tau), 0);            
            Hbar = 2 * H - Hold;
            Hold = H;
        end

        % update W
        sigma = sqrt(n/rank) * sum(H(:)) ./ sum(V,2)  / norm(H);
        tau   = sqrt(rank/n) * sum(V,2)  ./ sum(H(:)) / norm(H);

        for j = 1:options.inner_max_epoch
            % chi  = chi + bsxfun(@times, Wbar*H, sigma);
            % chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
            % W    = max(W - bsxfun(@times, (chi + 1)*H', tau), 0);
            chi  = chi + Wbar * H .* sigma;
            chi  = (chi - sqrt(chi.^2 + V .* (4*sigma))) / 2;
            W    = max(W - ((chi + 1) * H' .* tau), 0);
            Wbar = 2 * W - Wold;
            Wold = W;
        end
        

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