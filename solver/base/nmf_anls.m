function [x, infos] = nmf_anls(V, rank, options)
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
%       options     : options
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
% Modified by H.Kasai on Apr. 04, 2017


    m = size(V, 1);
    n = size(V, 2); 
    
    if ~isfield(options, 'alg')
        alg = 'anls_asgroup';
    else
        if ~strcmp(options.alg, 'anls_asgroup') && ~strcmp(options.alg, 'anls_bpp') ...
                && ~strcmp(options.alg, 'anls_asgivens') 
            fprintf('Invalid algorithm: %s. Therfore, we use anls_asgroup (i.e., ANLS with Active Set Method and Column Grouping).\n', options.alg);
            alg = 'anls_asgroup';
        else
            alg = options.alg;
        end
    end     

    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end   
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end       

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end

    if ~isfield(options, 'x_init')
        W = rand(m, rank);
        H = rand(rank, n);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end 
    
    % initialize
    epoch = 0;    
    R = zeros(m, n);
    grad_calc_count = 0; 
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, W, H, R);
    infos.cost = f_val;
    optgap = f_val - f_opt;
    infos.optgap = optgap;   
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
    if verbose > 0
        fprintf('ANLS (%s): Epoch = 000, cost = %.16e, optgap = %.4e\n', alg, f_val, optgap); 
    end  
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end    

    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)           

        if strcmp(alg, 'anls_asgroup')
            ow = 0;
            H = nnlsm_activeset(W'*W, W'*V, ow, 1, H);
            W = nnlsm_activeset(H*H', H*V', ow, 1, W');
            W = W';
            
        elseif strcmp(alg, 'anls_asgivens')
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
            
        elseif strcmp(alg, 'anls_bpp')
            H = nnlsm_blockpivot(W'*W, W'*V, 1, H);
            W = nnlsm_blockpivot(H*H', H*V', 1, W');
            W = W';

        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % calculate cost and optgap 
        f_val = nmf_cost(V, W, H, R);
        optgap = f_val - f_opt;    
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos.epoch = [infos.epoch epoch];
        infos.cost = [infos.cost f_val];
        infos.optgap = [infos.optgap optgap];     
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % display infos
        if verbose > 0
            if ~mod(epoch, disp_freq)
                fprintf('ANLS (%s): Epoch = %03d, cost = %.16e, optgap = %.4e\n', alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', max_epoch);
    end 
    
    x.W = W;
    x.H = H;

end