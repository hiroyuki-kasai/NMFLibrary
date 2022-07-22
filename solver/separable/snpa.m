function [x, infos] = snpa(V, num_col, in_options)
% Successive Nonnegative Projection Algorithm (variant with f(.) = ||.||^2)
%
%       At each step of the algorithm, the column of X maximizing ||.||_2 is 
%       extracted, and X is updated with the residual of the projection of its 
%       columns onto the convex hull of the columns extracted so far. 
%
% Inputs:
%       matrix      V
%       num_col     number of columns to be extracted.
%       options     options
%           normalization: 1: scale the columns of X so that they sum to one,
%                           hence matrix H will satisfy the assumption above for any
%                           nonnegative separable matrix X. 
%                          0: the default value for which no scaling is
%                           performed. For example, in hyperspectral imaging, this 
%                           assumption is already satisfied and normalization is not
%                           necessary.
%
% Output:
%       w           solution of w
%           K        : index set of the extracted columns. 
%           H        : optimal weights, that is, H = argmin_{Y >= 0} ||X-X(:,K)Y||_F
%       infos       information
%
% References:
%       N. Gillis, 
%       "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation," 
%       SIAM J. on Imaging Sciences 7 (2), 
%       pp. 1420-1450, 2014.
%    
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%   SNPA.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%   by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%   June. 21, 2022 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.disp_freq = 1;    
    local_options.normalize = 0;
    local_options.relerr = 1e-6;
    local_options.inner_max_epoch = 500;
    local_options.inner_nnls_alg = 'fpgm';
    local_options.special_stop_condition = @(epoch, infos, options, stop_options) spna_stop_func(epoch, infos, options, stop_options);       
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end      
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    % initialize
    method_name = 'SNPA';    
    i = 0; 
    grad_calc_count = 0;
    stop_options = [];        

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % initialize for this algorithm
    options.max_epoch = num_col+1;    
    if options.normalize == 1
        % normalize the columns of V of which colum is L1-norm = 1
        V = normalize_W(V, 1);
    end

    U = zeros(m, num_col);
    K = zeros(1, num_col);
    H = zeros(num_col, n);    
    normV0 = sum(V.^2); 
    nVmax = max(normV0); 
    normR = normV0; 
    VtUK = []; 
    UKtUK = [];

    % set for nnls subsolver
    nnls_options = [];
    nnls_options.verbose = 0;
    nnls_options.inner_max_epoch = options.inner_max_epoch;
    nnls_options.algo = options.inner_nnls_alg;    
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, eye(m), V, [], options, [], i, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
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
        
        % select the column of the residual R with largest l2-norm
        [a, ~] = max(normR); 
        
        % check ties up to 1e-6 precision
        b = find((a-normR)/a <= 1e-6); 
        
        % In case of a tie, select column with largest norm of the input matrix X 
        if length(b) > 1
            [~, d] = max(normX0(b)); 
            b = b(d); 
        end
        
        % update the index set, and extracted column
        K(i) = b; 
        U(:, i) = V(:, b); 
        
        % update MtUJ
        VtUK = [VtUK, V' * U(:, i)]; 
        
        % update UJtUJ
        if i == 1
            UtUi = [];
        else
            UtUi = U(:, 1:i-1)' * U(:, i); 
        end 
        UKtUK = [UKtUK, UtUi ; UtUi', U(:, i)' * U(:, i)]; 
        
        % update residual 
        if i == 1
            % Fast gradient method for min_{y in Delta} ||M(:, i)-M(:,J)y||
            [H, ~, ~, ~] = nnls_solver(V, V(:, K(1:i)), nnls_options);   
        else
            H(:, K(i)) = 0; 
            h = zeros(1,n); h(K(i)) = 1; 
            H = [H; h]; 
            nnls_options.init = H; 
            [H, ~, ~, ~] = nnls_solver(V, V(:, K(1:i)), nnls_options);               
        end
        
        % update the norm of the columns of the residual without computing it explicitely. 
        if i == 1
            normR = normV0 - 2 * ((VtUK') .* H) + (H .* (UKtUK*H));
        else
            normR = normV0 - 2 * sum((VtUK') .* H) + sum(H .* (UKtUK*H));
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;     

        % update epoch
        i = i + 1;        
        
        % store info
        infos = store_nmf_info(V, U(:, 1:i-1), H, [], options, infos, i-1, grad_calc_count, elapsed_time);          
        
        % display info
        display_info(method_name, i-1, infos, options);

    end
    
    x.K = K;
    x.U = U;
    x.H = H;   

end


function [stop_flag, reason, rev_infos] = spna_stop_func(epoch, infos, options, stop_options)

    stop_flag = false;
    reason = [];
    rev_infos = [];
  
    normR = stop_options.normR;
    nVmax = stop_options.nVmax;

    if sqrt(max(normR)/nVmax) < options.relerr 
        stop_flag = true;
        reason = sprintf('precision reached: sqrt(max(normR)/nVmax) = %.4e < options.relerr = %.4e\n', sqrt(max(normR)/nVmax), options.relerr);
        return;        
    end

end