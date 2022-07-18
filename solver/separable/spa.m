function [x, infos] = spa(V, num_col, in_options)
% Successive projection algorithm for separable NMF (SPA-NMF)
%
% The problem of interest is defined as
%
%       min ||V - UU^T V||_F^2,
%       where 
%       U is orthogonal, i.e., U^T U = I_num_col. 
%
%       At each step of the algorithm, the column of R maximizing ||.||_2 is 
%       extracted, and R is updated by projecting its columns onto the orthogonal 
%       complement of the extracted column. The residual R is initializd with X.
%
%
% Inputs:
%       matrix      V
%                   Ideally admitting a near-separable factorization, that is, 
%                   X = WH + N, where 
%                   conv([W, 0]) has r vertices,  
%                   H = [I,H']P where I is the identity matrix, H'>= 0 and its 
%                   columns sum to at most one, P is a permutation matrix, and
%                   N is sufficiently small. 
%       num_col     number of columns to be extracted. 
%           
% Output:
%       w.K         index set of the extracted columns. 
%       w.U         orthogonal matrix
%       infos       information
%
% References:
%       N. Gillis and S.A. Vavasis, 
%       "Fast and Robust Recursive Algorithms for Separable Nonnegative Matrix Factorization,"  
%       IEEE Trans. on Pattern Analysis and Machine Intelligence 36 (4), pp. 698-714, 
%       2014.
%
%       This implementation of the algorithm is based on the formula 
%       ||(I-uu^T)v||^2 = ||v||^2 - (u^T v)^2
%       which allows to avoid the explicit computation of the residual
%
%
% This file is part of NMFLibrary.
%
%       This file has been ported from 
%       SPA.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 14, 2022 (Hiroyuki Kasai): Ported initial version 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.disp_freq = 1;
    local_options.normalize = false;
    local_options.precision = 1e-6;
    local_options.special_nmf_cost = @(V, W, H, R, options) spa_nmf_cost_func(V, W, H, R, options); 
    local_options.special_stop_condition = @(epoch, infos, options, stop_options) spa_stop_func(epoch, infos, options, stop_options);    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
      
    % initialize
    method_name = 'SPA';
    i = 0; 
    grad_calc_count = 0;
    stop_options = [];    

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end       

    % initialize for this algorithm
    options.max_epoch = num_col + 1;
    if options.normalize == 1
        % normalize the columns of V of which colum is L1-norm = 1
        V = normalize_W(V, 1);
    end  

    U = zeros(m, num_col);
    K = zeros(1, num_col);
    normV0 = sum(V.^2); 
    nVmax = max(normV0); 
    normR = normV0; 
    
    % store initial info
    clear infos;
    options.r = 0;
    [infos, f_val, optgap] = store_nmf_info(V, [], [], [], options, [], i, grad_calc_count, 0);
    
    if options.verbose > 1
        %fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();

    % main loop
    i = i + 1;
    while true

        % check stop condition
        stop_options.normR = normR;
        stop_options.nVmax = nVmax;        
        [stop_flag, reason, max_reached_flag] = check_stop_condition(i, infos, options, stop_options);
        if stop_flag
            display_stop_reason(i, infos, options, method_name, reason, max_reached_flag);
            break;
        end
        
        % Select the column of M with largest l2-norm
        [a, b] = max(normR); 
        
        % Norm of the columns of the input matrix V 
        % Check ties up to 1e-6 precision
        b = find((a-normR)/a <= 1e-6); 
        
        % In case of a tie, select column with largest norm of the input matrix V 
        if length(b) > 1 
            [c, d] = max(normV0(b)); 
            b = b(d);
        end
        
        % update the index set, and extracted column
        K(i) = b; 
        U(:, i) = V(:, b); 
        
        % compute (I-u_{i-1}u_{i-1}^T)...(I-u_1u_1^T) U(:, i), that is, 
        % R^(i)(:,J(i)), where R^(i) is the ith residual (with R^(1) = M).
        for j = 1 : i-1
            U(:, i) = U(:, i) - U(:, j) * (U(:, j)' * U(:, i));
        end
        
        % normalize U(:, i)
        U(:, i) = U(:, i) / norm(U(:, i)); 
        
        % update the norm of the columns of V after orhogonal projection using
        % the formula ||r^(i)_k||^2 = ||r^(i-1)_k||^2 - ( U(:, i)^T V_k )^2 for all k. 
        normR = normR - (U(:, i)' * V).^2; 
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        i = i + 1;        
        
        % store info
        options.r = i-1;
        %options.U = U(:, 1:i-1);
        options.U = U;
        infos = store_nmf_info(V, [], [], [], options, infos, i-1, grad_calc_count, elapsed_time);          

        % display info
        display_info(method_name, i-1, infos, options);

    end

    x.K = K;
    x.U = U; 

end


% original cost function for SPA
% The latest version does not use this.
function res = spa_nmf_cost_func(V, W, H, R, options)
    res = 0;
    if options.r 
        res = norm(V - options.U * options.U' * V, 'fro')^2/2;
    else
        %
    end
end


function [stop_flag, reason, rev_infos] = spa_stop_func(epoch, infos, options, stop_options)

    stop_flag = false;
    reason = [];
    rev_infos = [];

    normR = stop_options.normR;
    nVmax = stop_options.nVmax;

    if sqrt(max(normR)/nVmax) < options.precision 
        stop_flag = true;
        reason = sprintf('precision reached: sqrt(max(normR)/nVmax) = %.4e < options.precision = %.4e\n', sqrt(max(normR)/nVmax), options.precision);
        return;        
    end

end