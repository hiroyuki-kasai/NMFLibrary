function [x, infos] = ns_nmf(V, rank, in_options)
% Nonsmooth nonnegative matrix factorization (Nonsmooth-NMF)
%
% The problem of interest is defined as
%
%      min || V - W*S*H ||_F^2,
%
%      or
%
%      min  D(V||W*S*H),
%
%      where 
%      {V, W, S, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, S, H} are calculated.
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
%       A. Pascual-Montano, J. M. Carazo, K. Kochi, D. Lehmann, and R. D. Pascual-Marqui, 
%       "Nonsmooth nonnegative matrix factorization (Nonsmooth-NMF),"
%       IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), vol.28, no.3, pp.403-415, 2006. 
%
%       Z. Yang, Y. Zhang, W. Yan, Y. Xiang, and S. Xie,
%       "A fast non-smooth nonnegative matrix factorization for learning sparse representation,"
%       IEEE Access, vol.4, pp.5161-5168, 2016.
%
%
% This file is part of NMFLibrary.
%
% This file is originally created by Graham Grindlay.
%
% 2010-01-14 Graham Grindlay (grindlay@ee.columbia.edu)
%
% Copyright (C) 2008-2010 Graham Grindlay (grindlay@ee.columbia.edu)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
% This file is partially created by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014
%
%
% Created by modifiying the original code by H.Kasai on Jul. 23, 2018 
%
% Change log: 
%
%       Jul. 29, 2019 (Hiroyuki Kasai): Modified.
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jun. 24, 2022 (Hiroyuki Kasai): Fixed a bug of W normalization.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options 
    local_options.theta         = 0.5; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    local_options.metric_type   = 'euc'; % 'euc' (default) or 'kl-div'
    local_options.update_alg    = 'mu';  % 'mu' or 'apg'
    local_options.apg_maxiter   = 100;
    local_options.myeps         = 1e-16;
    local_options.norm_w        = 1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  

    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;     

    % initialize
    method_name = 'Nonsmooth-NMF';      
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm    
    I = eye(rank);
    S = (1-options.theta) * I + (options.theta/rank) * ones(rank);
    
    % store initial info
    clear infos;
    WS = W * S;    
    [infos, f_val, optgap] = store_nmf_info(V, WS, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method, f_val, optgap); 
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

        if strcmp(options.update_alg, 'mu')
        
            % update H
            WS = W * S;
            if strcmp(options.metric_type, 'euc')
                H = H .* (WS' * V) ./ ((WS' * WS) * H + 1e-9);
            elseif strcmp(options.metric_type, 'kl-div')
                H = H .* (WS' * (V./(WS * H + 1e-9))) ./ (sum(WS, 1)' * ones(1,n));
            else
                error('Invalid metric type')
            end  

            % normalize rows in H
            [W, H] = normalize_WH(V, W, H, rank, 'type2');

            % update W
            SH = S * H;
            if strcmp(options.metric_type, 'euc')
                W = W .* (V * SH') ./ (W * (SH * SH') + 1e-9);
            elseif strcmp(options.metric_type, 'kl-div')
                W = W .* ((V ./ (W * SH + 1e-9)) * SH') ./ (ones(m, 1) * sum(SH,2)');
            end  
            
        else % support only 'euc' metric.
            
            if 0
                % update H
                WS = W*S;
                [H, ~, ~] = nesterov_mnls_general(V, WS, [], H, 1, options.apg_maxiter, 'basic'); 


                % normalize rows in H
                [W, H] = normalize_WH(V, W, H, rank, 'type2');                

                % update W
                SH = S*H;
                [W, ~, ~] = nesterov_mnls_general(V, [], SH', W, 1, options.apg_maxiter, 'basic');
                
            else
                
                % update W
                SH = S * H;
                [W, ~, ~] = nesterov_mnls_general(V, [], SH', W, 1, options.apg_maxiter, 'basic');
                %W_prev = W;
                W = W + (W<options.myeps) .* options.myeps;
                
                % normalize W
                if options.norm_w
                    %W11 = bsxfun(@rdivide,W,sqrt(sum(W.^2,1)));
                    W = normalize_W(W, 2); 
                end
                
                % update H
                WS = W*S;
                [H, ~, ~] = nesterov_mnls_general(V, WS, [], H, 1, options.apg_maxiter, 'basic'); 
                
            end

        end

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        WS = W * S;
        infos = store_nmf_info(V, WS, H, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display info
        display_info(method_name, epoch, infos, options);           

    end     

    x.W = W;
    x.H = H;
    x.S = S;    
    
end