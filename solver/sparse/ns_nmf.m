function [x, infos] = ns_nmf(V, rank, in_options)
% Nonsmooth nonnegative matrix factorization (nsNMF)
%
% The problem of interest is defined as
%
%           min || V - W*S*H ||_F^2,
%
%           or
%
%           min  D(V||W*S*H),
%
%           where 
%           {V, W, S, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, S, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
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
% Reference:
%       A. Pascual-Montano, J. M. Carazo, K. Kochi, D. Lehmann, and R. D. Pascual-Marqui, 
%       "Nonsmooth nonnegative matrix factorization (nsNMF),"
%       IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), vol.28, no.3, pp.403-415, 2006. 
%
%       Z. Yang, Y. Zhang, W. Yan, Y. Xiang, and S. Xie,
%       "A fast non-smooth nonnegative matrix factorization for learning sparse representation,"
%       IEEE Access, vol.4, pp.5161-5168, 2016.
%
%
% Created by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014
% Created by modifiying the original code by H.Kasai on Jul. 23, 2018 
% Modified by H.Kasai on Jul. 29, 2018 
%
% Change log: 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options 
    local_options.theta         = 0.5; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    local_options.metric        = 'EUC'; % 'EUC' (default) or 'KL'
    local_options.update_alg    = 'mu';  % 'mu' or 'apg'
    local_options.apg_maxiter   = 100;
    local_options.myeps         = 1e-16;
    local_options.norm_w        = 1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    if options.verbose > 0
        fprintf('# nsNMF: started ...\n');           
    end     
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;     
    R = init_factors.R;

    I = eye(rank);
    S = (1-options.theta)*I + (options.theta/rank)*ones(rank);
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
    % store initial info
    clear infos;
    WS = W*S;    
    [infos, f_val, optgap] = store_nmf_infos(V, WS, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('nsNMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) 

        if strcmp(options.update_alg, 'mu')
        
            % update H
            WS = W*S;
            if strcmp(options.metric, 'EUC')
                H = H.*(WS'*V)./((WS'*WS)*H + 1e-9);
            elseif strcmp(options.metric, 'KL')
                H = H.*(WS'*(V./(WS*H + 1e-9)))./(sum(WS,1)'*ones(1,n));
            end  

            % normalize rows in H
            [W,H] = rowsum_R_one(W,H); % normalize rows in H

            % update W
            SH = S*H;
            if strcmp(options.metric, 'EUC')
                W = W.*(V*SH')./(W*(SH*SH') + 1e-9);
            elseif strcmp(options.metric, 'KL')
                W = W.*((V./(W*SH + 1e-9))*SH')./(ones(m,1)*sum(SH,2)');
            end  
            
        else % support only 'EUC' metric.
            
            if 0
                % update H
                WS = W*S;
                [H, ~, ~] = nesterov_mnls_general(V, WS, [], H, 1, options.apg_maxiter, 'basic'); 


                % normalize rows in H
                [W,H] = rowsum_R_one(W,H); % normalize rows in H

                % update W
                SH = S*H;
                [W, ~, ~] = nesterov_mnls_general(V, [], SH', W, 1, options.apg_maxiter, 'basic');
                
            else
                
                % update W
                SH = S*H;
                [W, ~, ~] = nesterov_mnls_general(V, [], SH', W, 1, options.apg_maxiter, 'basic');
                %W_prev = W;
                W = W + (W<options.myeps) .* options.myeps;
                
                % normalize W
                if options.norm_w
                    %W11 = bsxfun(@rdivide,W,sqrt(sum(W.^2,1)));
                    [W, ~] = data_normalization(W, [], 'std');
                end
                
                % update H
                WS = W*S;
                [H, ~, ~] = nesterov_mnls_general(V, WS, [], H, 1, options.apg_maxiter, 'basic'); 
                
            end

        end

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        WS = W*S;
        [infos, f_val, optgap] = store_nmf_infos(V, WS, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('nsNMF: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end      

    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# nsNMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# nsNMF: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end      

    x.W = W;
    x.H = H;
    x.S = S;    
end