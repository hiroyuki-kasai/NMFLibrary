function [x, infos] = div_admm_nmf(V, rank, in_options)
% Divergence-based ADMM algorithm for non-negative matrix factorization (KL-FPA-NMF).
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
%           d_beta  : parameter of beta divergence 
%                       (only beta=0 (IS) and beta=1 (KL) are supported)
%           rho     : smothing parameter
%           fixed   : vector containing the indices of the basis vectors in 
%                       W to hold fixed (e.g., when W is known a priori)
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       D.L. Sun and C. Fvotte, 
%       "Alternating direction method of multipliers for non-negative matrix 
%       factorization with the beta divergence," 
%       ICASSP 2014.
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
% Ported by M.Horie and H.Kasai on June 30, 2022
%
% Change log: 
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];   
    local_options.rho = 1;
    local_options.metric_type = 'beta-div';
    local_options.d_beta = 1; % parameter of beta divergence (only beta=0 (IS) and beta=1 (KL) are supported)
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);
    W = init_factors.W;
    H = init_factors.H;

    % initialize 
    method_name = sprintf('Div-ADMM (%s=%.1f)', options.metric_type, options.d_beta);    
    epoch = 0; 
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end  

    % initialize for this algorithm
    fixed=[]; 
    % get the vector of indices to update
    free = setdiff(1:rank, fixed);
    
    X = W*H;
    Wplus = W;
    Hplus = H;
    alphaX = zeros(size(X));
    alphaW = zeros(size(W));
    alphaH = zeros(size(H));  

    if options.d_beta == 0 && ~isfield(in_options, 'rho') % when IS divergence (d_beta == 0), rho should be much higher.
        local_options.rho = 1000;
    end
    
    % store initial info
    clear infos;

    %[options.metric_type, options.metric.param] = check_divergence(options);      
    
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
        H = (W'*W + eye(rank)) \ (W'*X + Hplus + 1/options.rho*(W'*alphaX - alphaH));
        
        % update W
        P = H*H' + eye(rank);
        Q = H*X' + Wplus' + 1/options.rho*(H*alphaX' - alphaW');
        W(:,free) = ( P(:,free) \ (Q - P(:,fixed)*W(:,fixed)') )';
        
        % update X (this is the only step that depends on beta)
        X_ap = W*H;
        if options.d_beta == 1

            b = options.rho*X_ap - alphaX - 1;
            X = (b + sqrt(b.^2 + 4*options.rho*V))/(2*options.rho);
            
        elseif options.d_beta == 0

            A = alphaX/options.rho - X_ap;
            B = 1/(3*options.rho) - A.^2/9;
            C = - A.^3/27 + A/(6*options.rho) + V/(2*options.rho);
            D = B.^3 + C.^2;

            X(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                A(D>=0)/3;

            phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
            X(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
            
        else
            error('beta is not currently supported.')
        end

        % update for H_+ and W_+
        Hplus = max(H + 1/options.rho * alphaH, 0);
        Wplus = max(W + 1/options.rho * alphaW, 0);
        
        % update for dual variables
        alphaX = alphaX + options.rho * (X - X_ap);
        alphaH = alphaH + options.rho * (H - Hplus);
        alphaW = alphaW + options.rho * (W - Wplus);
        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;           
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);          
        
        % display info
        display_info(method_name, epoch, infos, options);
        
    end
    
    x.W = W;
    x.W(:,free) = Wplus(:,free);
    x.H = Hplus;

end