function [x, infos] = div_mu_nmf(V, rank, in_options)
% Divergence-based multiplicative upates (MU) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min f(V, W, H),
%       where 
%       {V, W, H} >= 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
%           alg     : mu: Multiplicative upates (MU)
%                       Reference for Euclidean distance and Kullback-Leibler divergence (kl-div):
%                           Daniel D. Lee and H. Sebastian Seung,
%                           "Algorithms for non-negative matrix factorization,"
%                           NIPS 2000. 
%
%                       Reference for Amari alpha divergence:
%                           A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
%                           "Extended SMART algorithms for non-negative matrix factorization,"
%                           Artificial Intelligence and Soft Computing, 2006.
%
%                           min D(V||R) = sum(V(:).^alpha .* R(:).^(1-d_alpha) - d_alpha*V(:) + (d_alpha-1)*R(:)) / (alpha*(d_alpha-1)), 
%                           where R = W*H.
%
%                           - Pearson's distance (d_alpha=2)
%                           - Hellinger's distance (d_alpha=0.5)
%                           - Neyman's chi-square distance (d_alpha=-1)
%
%                       Reference for beta divergence:
%                           A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
%                           "Extended SMART algorithms for non-negative matrix factorization,"
%                           Artificial Intelligence and Soft Computing, 2006.
%
%                           min D(V||W*H)
%
%                                               | sum(V(:).^d_beta + (d_beta-1)*R(:).^d_beta - ...
%                                               |     d_beta*V(:).*R(:).^(d_beta-1)) / ...
%                                               |     (d_beta*(d_beta-1))                  (d_beta \in{0 1}
%                          where D(V||R) =      |                                          
%                                               | sum(V(:).*log(V(:)./R(:)) - V(:) + R(:)) (d_beta=1) KL 
%                                               |
%                                               | sum(V(:)./R(:) - log(V(:)./R(:)) - 1)    (d_beta=0) IS
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
% This file is part of NMFLibrary
%
% Some part of this was originally created by G.Grindlay (grindlay@ee.columbia.edu) on Jan. 04, 2010.
%
% 2010-01-14 Graham Grindlay (grindlay@ee.columbia.edu)
%
% Copyright (C) 2008-2028 Graham Grindlay (grindlay@ee.columbia.edu)
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
% Created by H.Kasai on Feb. 16, 2017
%
% Change log: 
%
%       Jun. 27, 2022 (Hiroyuki Kasai): Separated from fro_mu_nmf.m (Frobenius norm MU).
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.metric_type   = 'kl-div'; % default    
    local_options.norm_h        = 0;
    local_options.norm_w        = 1;    
    %local_options.alpha         = 2;
    local_options.delta         = 0.1;
    local_options.d_alpha       = -1;   % for alpha divergence
    local_options.d_beta        = 0;    % for beta divergence 
    local_options.myeps         = 1e-16;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);        
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize 
    epoch = 0;    
    grad_calc_count = 0;
 
    % store initial info
    clear infos;
    options = check_divergence(options);
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    method_name = sprintf('Div-MU (%s=%.1f)', options.metric_type, options.metric_param);  

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
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

        if strcmp(options.metric_type, 'kl-div')
            
            % update W
            W = W .* ((V./(W*H + options.myeps))*H')./(ones(m,1)*sum(H'));
            if options.norm_w ~= 0
                W = normalize_W(W, options.norm_w);
            end                
            
            % update H
            H = H .* (W'*(V./(W*H + options.myeps)))./(sum(W)'*ones(1,n));
            if options.norm_h ~= 0
                H = normalize_H(H, options.norm_h);
            end                    

        elseif strcmp(options.metric_type, 'alpha-div')
            
            % update W
            W = W .* ( ((V+options.myeps) ./ (W*H+options.myeps)).^options.d_alpha * H').^(1/options.d_alpha);
            if options.norm_w ~= 0
                W = normalize_W(W, options.norm_w);
            end
            W = max(W, options.myeps);

            % update H
            H = H .* ( (W'*((V+options.myeps)./(W*H+options.myeps)).^options.d_alpha) ).^(1/options.d_alpha);
            if options.norm_h ~= 0
                H = normalize_H(H, options.norm_h);
            end
            H = max(H, options.myeps);
            
        elseif strcmp(options.metric_type, 'beta-div') || strcmp(options.metric_type, 'euc')
            
            WH = W * H;
            
            % update W
            W = W .* ( ((WH.^(options.d_beta-2) .* V)*H') ./ max(WH.^(options.d_beta-1)*H', options.myeps) );
            if options.norm_w ~= 0
                W = normalize_W(W, options.norm_w);
            end
            
            WH = W * H;

            % update H
            H = H .* ( (W'*(WH.^(options.d_beta-2) .* V)) ./ max(W'*WH.^(options.d_beta-1), options.myeps) );
            if options.norm_h ~= 0
                H = normalize_H(H, options.norm_h);
            end

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