function [x, infos] = admm_y_conv_nmf(V, rank, t, in_options)
% ADMM based convolutive non-negative matrix factorization (ADMM-Conv-NMF).
%
% The problem of interest is defined as
%
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       matrix      V
%       rank        rank
%           
% Output:
%       w           solution of w
%       infos       information
%
% References:
%
%    
% This file is part of NMFLibrary
%
% This file has been ported from 
% convNMF_MM1.m and convNMF_MM2.m at https://github.com/lyn202206/ADMM-Convolutive-NMF
% by Yinan Li.
%
% Ported by H.Kasai on June 29, 2022
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];    
    local_options.metric_type = 'beta-div';
    local_options.d_beta = 2;  
    local_options.rho = 1;
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end     
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);

    % initialize factors
    init_options = options;
    if ~isfield(options, 'x_init')
        W = zeros(m, rank, t);
        for i = 1 : t
            [init_factors, ~] = generate_init_factors(V, rank, init_options);    
            W(:, :, i) = init_factors.W;
        end
        H = init_factors.H;   
    else
        W = init_options.x_init.W;
        H = init_options.x_init.H;        
    end

    % initialize
    epoch = 0; 
    grad_calc_count = 0;

    options = check_divergence(options);
    sub_mode = sprintf('beta=%.1f', options.d_beta);
    if ~strcmp(options.metric_type, 'beta-div')
        sub_mode = options.metric_type;
    end 
    method_name = sprintf('ADMM-Y-Conv (%s)', sub_mode);    

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end

    % initialize for this algorithm
    X = zeros(m, n,t);
    for i=0:t-1
        tW = W(:, :, i+1);
        tH = shift_t(H, i);
        % initial X
        X(:, :, i+1) = tW * tH;
    end
    
    % some variable for test
    Xplus = X;
    U = zeros(m, n,t);
    
    alphaX = zeros(m, n,  t);
    alphaH = zeros(rank,n);
    alphaW = zeros(m, rank, t);
    
    Wplus = W;
    Hplus = H;

    % store initial info
    clear infos;

    [Wcon, Hcon] = reconstruct_wh(W, H, t);
    [infos, f_val, optgap] = store_nmf_info(V, Wcon, Hcon, [], options, [], epoch, grad_calc_count, 0);
      
    
    if options.verbose > 1
        fprintf('ADMM-Y-Conv (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', sub_mode, f_val, optgap); 
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

        % update H by accumulation
        P1 = zeros(rank, rank);
        P2 = zeros(rank, n);
        P3 = zeros(rank, n);
        for i = 0 : t-1
            tW = W(:, :, i+1);
            tX = shift_t(X(:, :, i+1),-i);
            talphaX = shift_t(alphaX(:, :, i+1), -i);
            P1 = P1 + tW'*tW;
            P2 = P2 + tW'*tX;
            P3 = P3 + tW'*talphaX;
        end
        H = (P1 + eye(rank)) \ (P2 + Hplus + 1 / options.rho * (P3 - alphaH));
      
        % update W
        for i = 0 : t-1
           % update W
           tH = shift_t(H, i);
           tX = X(:, :, i+1);
           P = tH*tH' + eye(rank);
           Q = tH*tX' + Wplus(:, :, i+1)' + 1 / options.rho*(tH*alphaX(:, :, i+1)' - alphaW(:, :, i+1)');
           W(:, :, i+1) = (P \ Q)';
        end
        
        %  update X_hat in each time slice
        for i = 0 : t-1
            tW = Wplus(:, :, i+1);
            tH = shift_t(Hplus, i);
            % initial X
            Xplus(:, :, i+1) = tW * tH;
        end
        % splite V in each time slice which result in U
        for i=0:t-1
            tW = Wplus(:, :, i+1);
            tH = shift_t(Hplus, i);
            U(:, :, i+1) = (tW * tH + eps).*V ./ sum(Xplus, 3);
        end
        
        if options.d_beta == 2
            % update for Euclidean distance
            for i=0:t-1
               tW = W(:, :, i+1);
               tH = shift_t(H, i);
               tV = U(:, :, i+1);
               % update X
               X(:, :, i+1) = (options.rho * tW * tH + tV - alphaX(:, :, i+1)) / (1 + options.rho);
            end
        elseif options.d_beta == 1
            % update for Kullback-Leibler divergence
            for i=0:t-1
                tW = W(:, :, i+1);
                tH = shift_t(H, i);
                tV = U(:, :, i+1);
                % update X
                b = options.rho * tW * tH - alphaX(:, :, i+1) - 1;
                X(:, :, i+1) = (b + sqrt(b.^2 + 4 * options.rho * tV)) / (2 * options.rho);                
            end
        elseif options.d_beta == 0
            % update for Itakura-Saito divergence
            for i=0:t-1
                tW = W(:, :, i+1);
                tH = shift_t(H, i);
                tV = U(:, :, i+1);
                % parameters for update
                A = alphaX(:, :, i+1)/options.rho - tW * tH;
                B = 1/(3*options.rho) - A.^2/9;
                C = - A.^3/27 + A / (6 * options.rho) + tV / (2*options.rho);
                D = B.^3 + C.^2;
                % update X
                tX = X(:, :, i+1);
                tX(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)), 3) + ...
                nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                A(D>=0)/3;
            
                phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
                tX(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
                X(:, :, i+1) = tX;
            end
        else       
            error('The options.d_beta you specified is not currently supported.')
        end
    
       %% ADMM update of Hplus, Wplus and Xplus
        Hplus = max(H + 1/options.rho*alphaH, 0);
        Wplus = max(W + 1/options.rho*alphaW, 0);
       
       % update for dual variables
        for i = 0 : t-1
            tW = W(:, :, i+1);
            tH = shift_t(H, i);
            % update alphaX
            alphaX(:, :, i+1) = alphaX(:, :, i+1) +  options.rho * (X(:, :, i+1)- tW * tH);
        end
        alphaH = alphaH + options.rho*(H - Hplus);
        alphaW = alphaW + options.rho*(W - Wplus);        

        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        [Wcon, Hcon] = reconstruct_wh(W, H, t);        
        infos = store_nmf_info(V, Wcon, Hcon, [], options, infos, epoch, grad_calc_count, elapsed_time);          
       
        % display info
        display_info(method_name, epoch, infos, options);
        
    end

    W = Wplus;
    H = Hplus;
    [W, H] = renormalize_convNMF(W, H);      
    
    x.W = W;
    x.H = H;    

end


function [W_concat, H_concat] = reconstruct_wh(W, H, t)
    
    W_concat = [];
    H_concat = [];  
    for j = 1 : t
        W_concat = [W_concat W(:, :,j)];
        H_concat = [H_concat; shift_t(H, j-1)]; 
    end 
end