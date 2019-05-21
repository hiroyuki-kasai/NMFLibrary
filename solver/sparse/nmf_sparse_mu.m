function [x, infos] = nmf_sparse_mu(V, rank, in_options)
% Sparse Multiplicative upates (MU) for non-negative matrix factorization with orthogonalization (Orth-NMF).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2 + lambda*sum(sum(H)),
%
%           or
%
%           min  D(V||W*H) + lambda*sum(sum(H)),
%
%           where 
%           {V, W, H} >= 0, and H is sparse.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
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
% References
%       D. Lee and S. Seung, 
%       "Algorithms for non-negative matrix factorization", 
%       NIPS, 2001.
%
%       J. Eggert and E. Korner, 
%       "Sparse coding and NMF", 
%        IEEE International Joint Conference on Neural Networks, 2004.
%
%       M. Schmidt, J. Larsen, and F. Hsiao, 
%       "Wind noise reduction using non-negative sparse coding", 
%       IEEE Workshop on Machine Learning for Signal Processing (MLSP), 2007.
%   
%
% Originally created by G.Grindlay (grindlay@ee.columbia.edu) on Jan. 14, 2010
% Modified by H.Kasai on Jul. 23, 2018
%
% Change log: 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options
    local_options.norm_h    = 0;
    local_options.norm_w    = 1;
    local_options.lambda    = 0.1;
    local_options.myeps     = 1e-16;
    local_options.metric    = 'EUC'; % 'EUC' (default) or 'KL'    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    if options.verbose > 0
        fprintf('# Sparse-MU: started ...\n');           
    end     
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    R = init_factors.R;
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    
    % preallocate matrix of ones
    if strcmp(options.metric, 'EUC')
        Omm = ones(m, m);   
    elseif strcmp(options.metric, 'KL')    
        Omm = ones(m, m);
        Omn = ones(m, n);
    end
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
   
    % store initial info
    clear infos;
    metric_param = [];
    if strcmp(options.metric, 'ALPHA-D')
        metric_param = options.alpha;
    elseif strcmp(options.metric, 'BETA-D')
        metric_param = options.beta;        
    end
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, [], epoch, grad_calc_count, 0, options.metric, metric_param);
    % store additionally different cost
    reg_val = options.lambda*sum(sum(H));
    f_val_total = f_val + reg_val;
    infos.cost_reg = reg_val;
    infos.cost_total = f_val_total;      
    
    if options.verbose > 1
        fprintf('Sparse-MU: Epoch = 0000, cost = %.16e, cost-reg = %.16e, optgap = %.4e\n', f_val, reg_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        % update W
        if strcmp(options.metric, 'EUC')
            H = H .* ( (W'*V) ./ max((W'*W)*H + options.lambda, options.myeps) );
        elseif strcmp(options.metric, 'KL')
            H = H .* ( (W'*(V./(W*H))) ./ max(W'*Omn + options.lambda, options.myeps) );
        end
        if options.norm_h ~= 0
            H = normalize_H(H, options.norm_h);
        end

    
        % update W 
        if strcmp(options.metric, 'EUC')
            if options.norm_w == 1
                W = W .* ( (V*H' + (Omm * (W*(H*H') .* W) )) ./ ...
                           max(W*(H*H') + (Omm*V*H' .* W), options.myeps) );
            elseif options.norm_w == 2
                W = W .* ( (V*H' + W .* (Omm * (W*(H*H') .* W) )) ./ ...
                           max(W*(H*H') + W .* (Omm*V*H' .* W), options.myeps) );
            end
        elseif strcmp(options.metric, 'KL')
        
            C = V./(W*H);
            if options.norm_w == 1
                W = W .* ( (C*H' + (Omm*(Omn*H' .* W))) ./ ...
                           max(Omn*H' + (Omm*(C*H' .* W)), options.myeps) );
            elseif options.norm_w == 2
                W = W .* ( (C*H' + W .* (Omm*(Omn*H' .* W))) ./ ...
                           max(Omn*H' + W .* (Omm*(C*H' .* W)), options.myeps) );
            end
        end
        W = normalize_W(W, options.norm_w);
        

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time, options.metric, metric_param);  
        % store additionally different cost
        reg_val = options.lambda*sum(sum(H));
        f_val_total = f_val + reg_val;
        infos.cost_reg = [infos.cost_reg reg_val];
        infos.cost_total = [infos.cost_total f_val_total];        
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('Sparse-MU: Epoch = %04d, cost = %.16e, cost-reg = %.16e, optgap = %.4e\n', epoch, f_val, reg_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# Sparse-MU: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# Sparse-MU: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;
end
