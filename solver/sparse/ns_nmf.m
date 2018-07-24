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
%
% Created by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014
% Modified by H.Kasai on Jul. 23, 2018


    % set dimensions and samples
    m = size(V, 1);
    n = size(V, 2);

    % set local options 
    local_options.theta     = 0; % decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
    local_options.metric    = 'EUC'; % 'EUC' (default) or 'KL'
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    

    I = eye(rank);
    S = (1-options.theta)*I + (options.theta/rank)*ones(rank);
    
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0; 
    
    if ~isfield(options, 'x_init')
        [W, H] = NNDSVD(abs(V), rank, 0);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end    
      

    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
   
    % store initial info
    clear infos;
    WS = W*S;    
    [infos, f_val, optgap] = store_nmf_infos(V, WS, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('nsNMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) 

        %do_updates();
        
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

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        WS = W*S;
        [infos, f_val, optgap] = store_nmf_infos(V, WS, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        
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
end