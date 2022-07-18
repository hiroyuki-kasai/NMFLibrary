function [x, infos] = symm_anls(V, rank, in_options)
% Symmetric non-negative matrix factorization by ANLS (Symm ANLS).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2 + alpha * ||W-H'||_F^2,
%       where 
%       {V, W, H} >= 0, and W is close to H'.
%
% Given a symmetric non-negative matrix V, factorized non-negative matrices {W, H(close to W')} are calculated.
%
%
% Inputs:
%       V           : (m x m) symmetric non-negative matrix to factorize
%       rank        : rank
%       in_options 
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x m)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
% References
%       D. Kang, C. Ding, H. Park,
%       "Symmetric Nonnegative Matrix Factorization for Graph Clustering,"
%       The 12th SIAM International Conference on Data Mining (SDM'12), pp.106-117, 2012.
%
%       D. Kuang, S. Yun, H. Park,
%       "SymNMF Nonnegative low-rank approximation of a similarity matrix for graph clustering,"
%       Journal of Global Optimization, vol.62, no.3, pp.545-574, 2015.
%
%       Z. Zhu, X. Li, K. Liu, Q. Li, 
%       "Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization,"
%       NIPS, 2018.
%
%
% This file is part of NMFLibrary.
%
% Originally created by Da Kuang et al.
%       See https://github.com/dakuang/symnmf
%       See https://github.com/lixiao0982/Dropping-Symmetric-for-Symmetric-NMF.
%
% Modified by H.Kasai on June 24, 2019 for NMFLibrary
%
% Change log: 
%
%       Jul. 14, 2022 (Hiroyuki Kasai): Fixed algorithm.
%

    
    % set dimensions and samples
    [m] = size(V, 1);
 
    % set local options
    local_options = [];
    local_options.alpha = max(max(V))^2;
    local_options.init_alg  = 'symm_mean';
    local_options.use_kuang_code = true;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = (init_factors.H)';      

    % initialize
    method_name = 'Symm (ANLS)';    
    epoch = 0;    
    grad_calc_count = 0; 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end       

    % initialize for this algorithm
    I_k = options.alpha * eye(rank);
    left = H' * H;
    right = V * H;     
    
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H', [], options, [], epoch, grad_calc_count, 0);
    
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
        
        if options.use_kuang_code % use Kuang's original code
            W = nnlsm_blockpivot(left + I_k, (right + options.alpha * H)', 1, W')';
            left = W' * W;
            right = V * W;
            H = nnlsm_blockpivot(left + I_k, (right + options.alpha * W)', 1, H')';
            left = H' * H;
            right = V * H;

            if options.alpha == 0
                norms_W = sum(W.^2) .^ 0.5;
                norms_H = sum(H.^2) .^ 0.5;
                norms = sqrt(norms_W .* norms_H);
                W_tmp = bsxfun(@times, W, norms./norms_W);
                H_tmp = bsxfun(@times, H, norms./norms_H);
                H_tmp = H_tmp';
            else
                W_tmp = W;
                H_tmp = H';
            end 
        else
            % update U
            Wt = nnlsm_blockpivot([H;sqrt(options.alpha)*eye(rank)], [V';sqrt(options.alpha)*H'], 0, W');
            W = Wt';
            W_tmp = W;
            
            % update V
            Ht = nnlsm_blockpivot([W_tmp;sqrt(options.alpha)*eye(rank)], [V;sqrt(options.alpha)*W_tmp'], 0, W');
            H = Ht';
            H_tmp = Ht;
        end
      

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*m;        
        
        % update epoch
        epoch = epoch + 1;         

        % store info
        infos = store_nmf_info(V, W_tmp, H_tmp, [], options, infos, epoch, grad_calc_count, elapsed_time);    

        % display info
        display_info(method_name, epoch, infos, options);            

    end    

    if options.use_kuang_code && options.alpha == 0
        norms_W = sum(W.^2) .^ 0.5;
        norms_H = sum(H.^2) .^ 0.5;
        norms = sqrt(norms_W .* norms_H);
        W = bsxfun(@times, W, norms./norms_W);
        H = bsxfun(@times, H, norms./norms_H);
    end

    x.W = H;
    x.H = H';

end