function [x, infos] = convex_mu_nmf(V, rank, in_options)
% (Kernel) Convex multiplicative update for non-negative matrix factorization ((Kernel-)Convex-MU-NMF).
%
% The problem of interest is defined as
%
%  (standard)   min || VWH - V ||_F^2,
%               where 
%               {W, H} >= 0, and 
%
%  (kernel)     min || K(V,V) WH - K(V,V) ||_F^2,
%               where 
%               {W, H} >= 0, and K(V,V) is a kernel matrix. 
%
%
% Given a V with mixed-signs, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       matrix      V
%       rank        rank
%       in_options 
%           sub_mode: 'std' (default) or 'kernel' 
%           kernel  : kernel type (rbf (default), polynomial, linear, sigmoid)
%           
% Output:
%       x           solution of x
%       infos       information
%
% References:
%       C. Ding, T. Li, and M.I. Jordan,
%       "Convex and semi-nonnegative matrix factorizations,"
%       IEEE Transations on Pattern Analysis and Machine Intelligence,
%       vol. 32, no. 1, pp. 45-55, 2010.
%
%       T. Li and C. Ding,
%       "The Relationships Among Various Nonnegative Matrix Factorization Methods for Clustering,"
%       International Conference on Data Mining,
%       2006.
%
%       Y. Li and A. Ngom,
%       "A New Kernel Non-Negative Matrix Factorization and Its Application in Microarray Data Analysis,"
%       CIBCB,
%       2012.
%    
%
% This file is part of NMFLibrary
%
% This file has been ported from 
% convexnmfrule.m and kernelconvexnmf.m at https://sites.google.com/site/nmftool/home/source-code
% by Yifeng Li.
%
%%%%
% Copyright (C) <2012>  <Yifeng Li>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 01, 2011
%%%%
%
%
% This file was originally created by Graham Grindla.
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
% Ported by H.Kasai on June 30, 2022
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];  
    local_options.sub_mode = 'std';
    local_options.kernel ='rbf';  
    local_options.kernel_param = []; 
    local_options.special_nmf_cost = @(V, W, H, R, options) convex_mu_nmf_cost_func(V, W, H, R, options);
    local_options.special_init_factors = @(V, rank, wh_flag, r_flag, options) convex_mu_initialize(V, rank, wh_flag, r_flag, options);    
    
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
    epoch = 0; 
    grad_calc_count = 0;

    % initialize for this algorithm
    if strcmp(options.sub_mode, 'std')
        Ak = V' * V;
        X = V;  
        options.name = 'std';
    elseif strcmp(options.sub_mode, 'kernel')
        Ak = computeKernelMatrix(V, V, options);
        X = Ak;
        options.name = sprintf('kernel-%s', options.kernel);
    else
        error('Invalid sub_mode')
    end
    method_name = sprintf('Convex-MU (%s)', options.name);

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end    

    Ap = (abs(Ak) + Ak) ./ 2;
    An = (abs(Ak) - Ak) ./ 2;
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(X, W, H, [], options, [], epoch, grad_calc_count, 0);
      
    if options.verbose > 1
        fprintf('Convex-MU (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.name, f_val, optgap); 
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
        
        ApW = Ap * W;
        AnW = An * W;
        WH  = W * H;
   
        % update H
        H = H .* sqrt((ApW' + AnW' * WH) ./ (AnW' + ApW' * WH));
        H = max(H, eps);
        HHt = H * H';

        % update W
        W = W .* sqrt((Ap * H' + AnW * HHt) ./ (An * H' + ApW * HHt)); 
        W = max(W, eps);

        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info    
        infos = store_nmf_info(X, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);          
     
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;

end


function res = convex_mu_nmf_cost_func(V, W, H, R, options)

    res = norm(V - V * W * H, 'fro');

end


function [init_factors, init_factors_opts] = convex_mu_initialize(V, rank, wh_flag, r_flag, options)

    init_factors_opts = [];

    [~, n] = size(V);

    if wh_flag
        H = rand(rank, n);
        W = H' * diag(1./sum(H,2)');
    else
        W = options.x_init.W;
        H = options.x_init.H;       
    end

    init_factors.W = W;
    init_factors.H = H;   
    init_factors.R = zeros(size(V));

end