function [x, infos] = orth_mu_nmf(V, rank, in_options)
% Orthogonal multiplicative upates (MU) for non-negative matrix factorization (Orth-NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} >= 0, and W or H is orthogonal.
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
% References
%       S. Choi, "Algorithms for orthogonal nonnegative matrix factorization",
%       IEEE International Joint Conference on Neural Networks, 2008.
%       D. Lee and S. Seung, "Algorithms for Non-negative Matrix Factorization", 
%       NIPS, 2001.
%   
%
% This file is part of NMFLibrary.
%
% Originally created by G.Grindlay (grindlay@ee.columbia.edu) on Nov. 04, 2010.
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
% Ported by H.Kasai on Jul. 23, 2018
%
% Change log: 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.orth_h    = 1;
    local_options.norm_h    = 1;
    local_options.orth_w    = 0;
    local_options.norm_w    = 0;
    local_options.myeps     = 1e-16;
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end      
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);   
    
    % check
    if ~(options.norm_w && options.orth_w) && ~(options.norm_h && options.orth_h)
        warning('nmf_euc_orth: orthogonality constraints should be used with normalization on the same mode!');
    end    
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    
    % initialize
    method_name = 'Orth-MU';
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    if options.orth_h || options.orth_w
        if options.orth_h
            orth_val = norm(H*H' - eye(rank),'fro');
        else
            orth_val = norm(W'*W - eye(rank),'fro');
        end
        [infos.orth] = orth_val;
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

        % update W        
        if options.orth_w
            W = W .* ( (V*H') ./ max(W*H*V'*W, options.myeps) );
        else
            W = W .* ( (V*H') ./ max(W*(H*H'), options.myeps) );
        end
        if options.norm_w ~= 0
            W = normalize_W(W, options.norm_w);
        end

        % update H
        if options.orth_h
            H = H .* ( (W'*V) ./ max(H*V'*W*H, options.myeps) );
        else
            H = H .* ( (W'*V) ./ max((W'*W)*H, options.myeps) );
        end
        if options.norm_h ~= 0
            H = normalize_H(H, options.norm_h);
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        if options.orth_h || options.orth_w
            if options.orth_h
                orth_val = norm(H*H' - eye(rank), 'fro');
            else
                orth_val = norm(W'*W - eye(rank), 'fro');
            end
            [infos.orth] = [infos.orth orth_val];
        end
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;

end