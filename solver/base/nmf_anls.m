function [x, infos] = nmf_anls(V, rank, in_options)
% Alternative non-negative least squares (ANLS) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2,
%           where 
%           {V, W, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options
%
%
% References:
%       Jingu Kim, Yunlong He, and Haesun Park,
%       "Algorithms for Nonnegative Matrix and Tensor Factorizations: A Unified View 
%       Based on Block Coordinate Descent Framework,"
%       Journal of Global Optimization, 58(2), pp. 285-319, 2014.
%
%       Jingu Kim and Haesun Park.
%       "Fast Nonnegative Matrix Factorization: An Active-set-like Method and Comparisons,"
%       SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
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
% This code calls functions {nnls1_asgivens, nnlsm_activeset, nnlsm_blockpivot}
% written by Jingu Kim. See https://github.com/kimjingu/nonnegfac-matlab.
%
% Created by H.Kasai on Apr. 04, 2017
% Modified by H.Kasai on Oct. 27, 2017
%
% Change log: 
%
%   Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.options.alg   = 'anls_asgroup';

    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);    

    % set paramters
    if ~strcmp(options.alg, 'anls_asgroup') && ~strcmp(options.alg, 'anls_asgivens') ...
            && ~strcmp(options.alg, 'anls_bpp') 
        fprintf('Invalid algorithm: %s. Therfore, we use anls_asgroup (i.e., ANLS with Active Set Method and Column Grouping).\n', options.alg);
        options.alg = 'anls_asgroup';
    end
    
    if options.verbose > 0
        fprintf('# ANLS (%s): started ...\n', options.alg);           
    end  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0;
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('ANLS (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, f_val, optgap); 
    end  
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);    

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        if strcmp(options.alg, 'anls_asgroup')
            ow = 0;
            H = nnlsm_activeset(W'*W, W'*V, ow, 1, H);
            W = nnlsm_activeset(H*H', H*V', ow, 1, W');
            W = W';
            
        elseif strcmp(options.alg, 'anls_asgivens')
            ow = 0;
            WtV = W' * V;
            for i=1:size(H,2)
                H(:,i) = nnls1_asgivens(W'*W, WtV(:,i), ow, 1, H(:,i));
            end

            HAt = H*V';
            Wt = W';
            for i=1:size(W,1)
                Wt(:,i) = nnls1_asgivens(H*H', HAt(:,i), ow, 1, Wt(:,i));
            end
            W = Wt';
            
        elseif strcmp(options.alg, 'anls_bpp')
            H = nnlsm_blockpivot(W'*W, W'*V, 1, H);
            W = nnlsm_blockpivot(H*H', H*V', 1, W');
            W = W';

        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('ANLS (%s): Epoch = %04d, cost = %.16e, optgap = %.4e\n', options.alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# ANLS (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg, f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# ANLS (%s): Max epoch reached (%g).\n', options.alg, options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;

end