function [x, infos] = semi_bcd_nmf(V, rank, in_options)
% Semi non-negative matrix factorization (Semi-NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       H > 0.
%
%       If the optimal rank-r approximation of X is semi-nonnegative, then the
%       code returns an optimal solution. Otherwise, it uses block coordiante 
%       descent for semi-NMF initialized with the SVD. 
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options    
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
% Reference:
%       N. Gillis and A. Kumar, 
%       "Exact and Heuristic Algorithms for Semi-Nonnegative Matrix Factorization", 
%       SIAM J. on Matrix Analysis and Applications 36 (4), pp. 1404-1424, 
%       2015. 
%
%
% This file is part of NMFLibrary.
%
%       This file has been ported from 
%       semiNMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 15, 2022 (Hiroyuki Kasai): Ported initial version 
%

    
    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.use_seminmf_init = true;
    local_options.inner_max_epoch = 500;
    local_options.inner_nnls_alg = 'hals';
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end      
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    method_name = 'Semi-BCD';     
    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end  
    
    % initialize factors
    clear infos;

    if numel(V) < 1e8 && ~issparse(V)
        [u,s,v] = svds(V, rank); 
        Vr = u * s * v'; 
        [seminnrank, W, H] = seminonnegativerank(Vr); 
    else
        seminnrank = 0; 
    end

    % if the truncated SVD is semi-nonnegative --> we can compute an optimal solution
    if seminnrank == rank % cf. Corollary 3.3 in the paper above.
        [infos, f_val, ~] = store_nmf_info(V, u*s, v', [], options, [], 0, 0, 0);
        if options.verbose > 1
            fprintf('Semi-BCD: truncated SVD is semi-nonnegative. cost = %.16e\n', f_val); 
        end
        x.W = u*s;
        x.H = v';        
        return;
    end

    init_options = options;    
    if options.use_seminmf_init
        [W, H] = SVDinitSemiNMF(V, rank);
    else
        [init_factors, ~] = generate_init_factors(V, rank, init_options);    
        W = init_factors.W;
        H = init_factors.H;      
    end
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0;
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('Semi-BCD: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
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
        W = V / H;
        
        % update H using nnls_solver
        nnls_options.init = H; 
        nnls_options.verbose = 0;
        nnls_options.inner_max_epoch = options.inner_max_epoch;
        nnls_options.algo = options.inner_nnls_alg;
        [H, ~, ~] = nnls_solver(V, W, nnls_options); % BCD on the rows of H       
        
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