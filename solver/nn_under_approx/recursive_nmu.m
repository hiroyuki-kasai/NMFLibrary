function [x, infos] = recursive_nmu(V, rank, in_options)
% Recursive non-negative matrix underapproximation (Recursive-NMU).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} > 0 and WH <= V.
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
%           Cnorm       Choice of the norm 1 or 2, default = 2.
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       N. Gillis and F. Glineur,
%       "Using Underapproximations for Sparse Nonnegative Matrix Factorization,"
%       Pattern Recognition 43 (4), pp. 1676-1687, 
%       2010.
%
%       N. Gillis and R.J. Plemmons,
%       "Dimensionality Reduction, Classification, and Spectral Mixture Analysis 
%       using Nonnegative Underapproximationm,"
%       Optical Engineering 50, 027001, 
%       2011.
%    
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%       recursiveNMU.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Ported by T.Fukunaga and H.Kasai on June 24, 2022 for NMFLibrary
%
% Change log: 
%
%


    % set dimensions and samples
    [m, n] = size(V);
    % set local options
    local_options = [];
    local_options.Cnorm = 2;
    local_options.inner_max_epoch = 200;
    
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
    M = V;

    % initialize
    method_name = 'Recursive-NMU';
    epoch = 0; 
    grad_calc_count = 0;
    
    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end        
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);   
    
    % initialize for this algorithm
    % (here)
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: k = 00, Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end     
         
    % set start time
    start_time = tic();
    prev_time = start_time;
    
    % main loop
    for k = 1 : rank
        
        % initialize epoch
        epoch = 0;
        
        % initialize (x,y) with an optimal rank-one NMF of M
        [w, s, h] = svds(M, 1);
        ws = abs(w) * sqrt(s);
        hs = abs(h) * sqrt(s);
        W(:, k) = ws;
        H(k, :) = hs'; 
        
        % initialize Lagrangian variable lambda
        R = M - ws * hs';
        lambda = max(zeros(size(R)), -R);
        
        % inner loop
        while (optgap > options.tol_optgap) && (epoch < options.inner_max_epoch) 

            % update ws and hs
            A = M - lambda;
            
            if options.Cnorm == 1
                % l_1 norm minimization
                ws = max(0, (wmedian(A, hs)));
                hs = max(0, (wmedian(A', ws)));
               
             elseif options.Cnorm == 2 
                % l_2 norm minimization 
                ws = max(0, A * hs);
                ws = ws / (max(ws) + 1e-16);
                hs = max(0, (A' * ws) / (ws' * ws));
            end
            
            % update lambda
            if sum(ws) ~= 0 && sum(hs) ~= 0
                R = M - ws * hs';
                W(:, k) = ws;
                H(k, :) = hs'; 
                lambda = max(0, lambda - R / ((epoch + 1) + 1));
            else
                lambda = lambda / 2;
                ws = W(:, k);
                hs = H(k, :)'; 
            end


            % measure elapsed time
            elapsed_time = toc(start_time); 
            
            % measure gradient calc count
            grad_calc_count = grad_calc_count + m*n;            

            % update epoch
            epoch = epoch + 1;

            % store info
            % total iteration is computed as (k - 1) * options.inner_max_epoch + epoch
            W_rec = W(:, 1:k);
            H_rec = H(1:k, :);            
            [infos, f_val, optgap] = store_nmf_info(V, W_rec, H_rec, [], options, infos, (k - 1) * options.inner_max_epoch + epoch, grad_calc_count, elapsed_time);          
    
            % display infos
            if options.verbose > 2
                if ~mod(epoch, disp_freq)
                    fprintf('%s: k = %02d, Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', method_name, k, (k - 1) * options.inner_max_epoch + epoch, f_val, optgap, elapsed_time - prev_time);
                end
            end              

        end

        M = max(0, M - ws * hs');

        % store info
        % total iteration is computed as (k - 1) * options.inner_max_epoch + epoch
        W_rec = W(:, 1:k);
        H_rec = H(1:k, :);            
        [infos, f_val, optgap] = store_nmf_info(V, W_rec, H_rec, [], options, infos, (k - 1) * options.inner_max_epoch + epoch, grad_calc_count, elapsed_time);          

        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('%s: k = %02d, Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', method_name, k, (k - 1) * options.inner_max_epoch + epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  

        prev_time = elapsed_time;

    end

    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# Recursive-NMU: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif (k - 1) * options.inner_max_epoch + epoch == rank * options.inner_max_epoch
            fprintf('# Recursive-NMU: Max epoch reached (%g).\n', rank * options.inner_max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;    
    
end

% WMEDIAN computes an optimal solution of
%
% min_x  || A - xy^T ||_1, y >= 0
%
% where A has dimension (m x n), x (m) and y (n),
% in O(mn log(n)) operations. Should be done in O(mn)...

function x = wmedian(A,y)

    % Reduce the problem for positive entries of y
    indi = y > 1e-16;
    A = A(:, indi);
    y = y(indi); 
    [m, n] = size(A);
    A = A ./ repmat(y', m, 1);
    y = y / sum(y);

    % Sort rows of A, m*O(n log(n)) operations
    [As, Inds] = sort(A, 2);

    % Construct matrix of ordered weigths
    Y = y(Inds);

    % Extract the median
    actind = 1 : m;
    i = 1; 
    sumY = zeros(m, 1);
    x = zeros(m, 1);
    while ~isempty(actind) % O(n) steps... * O(m) operations
        % sum the weitghs
        sumY(actind, :) = sumY(actind, :) + Y(actind, i);
        % check which weitgh >= 0
        supind = (sumY(actind, :) >= 0.5);
        % update corresponding x
        x(actind(supind)) = As(actind(supind), i);
        % only look reminding x to update
        actind = actind(~supind);
        i = i + 1;
    end
end
