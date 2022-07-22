function [x, infos] = symm_newton(V, rank, in_options)
% Symmetric non-negative matrix factorization by Newton's method (Symm Newton).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
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
%       Da Kuang, Chris Ding, Haesun Park,
%       "Symmetric Nonnegative Matrix Factorization for Graph Clustering,"
%       The 12th SIAM International Conference on Data Mining (SDM'12), pp.106-117, 2012.
%
%       Da Kuang, Sangwoon Yun, Haesun Park,
%       "SymNMF: Nonnegative low-rank approximation of a similarity matrix for graph clustering,"
%       Journal of Global Optimization, vol.62, no.3, pp.545-574, 2015.
%
%
% This file is part of NMFLibrary.
%
% Originally created by Da Kuang et al. See https://github.com/dakuang/symnmf
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
    local_options.sigma = 0.1;
    local_options.beta  = 0.1;
    local_options.init_alg  = 'symm_mean';    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    H = (init_factors.H)';      
    
    % initialize
    method_name = 'Symm (Newton)';
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     

    % initialize for this algorithm
    projnorm_idx = false(m, rank);
    R = cell(1, rank);
    p = zeros(1, rank);
    left = H'*H;  
    obj = norm(V, 'fro')^2 - 2 * trace(H' * (V*H)) + trace(left * left);    

    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, H, H', [], options, [], epoch, grad_calc_count, 0);
    
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
        
        gradH = 4*(H*(H'*H) - V*H);
        projnorm_idx_prev = projnorm_idx;
        projnorm_idx = gradH<=eps | H>eps;
        projnorm = norm(gradH(projnorm_idx));

        if mod(epoch, 100) == 0
            p = ones(1, rank);
        end

        step = zeros(m, rank);
        hessian = cell(1, rank);
        temp = H*H' - V;

        for i = 1 : rank
            if ~isempty(find(projnorm_idx_prev(:, i) ~= projnorm_idx(:, i), 1))
                hessian{i} = hessian_blkdiag(temp, H, i, projnorm_idx);
                [R{i}, p(i)] = chol(hessian{i});
            end
            if p(i) > 0
                step(:, i) = gradH(:, i);
            else
                step_temp = R{i}' \ gradH(projnorm_idx(:, i), i);
                step_temp = R{i} \ step_temp;
                step_part = zeros(m, 1);
                step_part(projnorm_idx(:, i)) = step_temp;
                step_part(step_part > -eps & H(:, i) <= eps) = 0;
                if sum(gradH(:, i) .* step_part) / norm(gradH(:, i)) / norm(step_part) <= eps
                    p(i) = 1;
                    step(:, i) = gradH(:, i);
                else
                    step(:, i) = step_part;
                end
            end
        end

        alpha_newton = 1;
        Hn = max(H - alpha_newton * step, 0);
        left = Hn'*Hn;
        newobj = norm(V, 'fro')^2 - 2 * trace(Hn' * (V*Hn)) + trace(left * left);
        if newobj - obj > options.sigma * sum(sum(gradH .* (Hn-H)))
            while true
                alpha_newton = alpha_newton * options.beta;
                Hn = max(H - alpha_newton * step, 0);
                left = Hn'*Hn;
                newobj = norm(V, 'fro')^2 - 2 * trace(Hn' * (V*Hn)) + trace(left * left);
                if newobj - obj <= options.sigma*sum(sum(gradH .* (Hn-H)))
                    H = Hn;
                    obj = newobj;
                    break;
                end
            end
        else
            H = Hn;
            obj = newobj;
        end
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*m;  
        
        % update epoch
        epoch = epoch + 1;         

        % store info
        infos = store_nmf_info(V, H, H', [], options, infos, epoch, grad_calc_count, elapsed_time);    

        % display info
        display_info(method_name, epoch, infos, options);    

    end

    x.W = H;
    x.H = H';    

end


function He = hessian_blkdiag(temp, H, idx, projnorm_idx)

    [n, k] = size(H);
    subset = find(projnorm_idx(:, idx) ~= 0); 
    hidx = H(subset, idx);
    eye0 = (H(:, idx)' * H(:, idx)) * eye(n);

    He = 4 * (temp(subset, subset) + hidx * hidx' + eye0(subset, subset));

end
