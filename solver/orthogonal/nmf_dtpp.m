function [x, infos] = nmf_dtpp(V, rank, in_options)
% Orthogonal multiplicative upates (MU) for non-negative matrix factorization (DTPP).
%
% The problem of interest is defined as
%
%           min || V - WSH ||_F^2,
%           where 
%           {V, W, H} >= 0, and . 
%           
%           W or H is orthogonal (Uni-orthogonal 2-factor NMF, with S (Identity matrix)), or
%           W and H are orthogonal (Bi-orthogonal 3-factor NMF).
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
% References
%       C.Ding, T.Li, W.Peng, and H.Park,
%       "Orthogonal nonnegative matrix t-factorizations for clustering",
%       12th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD), 
%       2006.
%   
%
% Created by H.Kasai on May 16, 2019
%
% Change log: 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.orth_h    = 1;
    local_options.norm_h    = 2;
    local_options.orth_w    = 0;
    local_options.norm_w    = 0;
    local_options.myeps     = 1e-16;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);   
    
    % check
%     if ~(options.norm_w && options.orth_w) && ~(options.norm_h && options.orth_h)
%         warning('nmf_euc_orth: orthogonality constraints should be used with normalization on the same mode!');
%     end  

    if options.verbose > 0
        fprintf('# DTPP: started ...\n');           
    end     
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    R = init_factors.R;
    
    if options.orth_w && options.orth_h
        S = pinv(W)*V*pinv(H); % Bi-orthogonal 3-factor NMF
    else
        S = eye(rank, rank);
    end      
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 

    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W*S, H, R, options, [], epoch, grad_calc_count, 0);
    if options.orth_h || options.orth_w
        if options.orth_h && ~options.orth_w
            orth_val = norm(H*H' - eye(rank), 'fro');
        elseif ~options.orth_h && options.orth_w
            orth_val = norm(W'*W - eye(rank), 'fro');
        else 
            orth_val_h = norm(H*H' - eye(rank), 'fro');
            orth_val_w = norm(W'*W - eye(rank), 'fro');
            orth_val = (orth_val_h + orth_val_w) / 2;
        end
        [infos.orth] = orth_val;
    end    

    if options.verbose > 1
        fprintf('DTPP: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        % update W        
        if options.orth_w
            W = W .* sqrt( (V*H'*S') ./ max(W*W'*V*H'*S', options.myeps) );
        else
            W = W .* ( (V*H') ./ max(W*(H*H'), options.myeps) );
        end
        if options.norm_w ~= 0
            W = normalize_W(W, options.norm_w);
        end

        % update H
        if options.orth_h
            H = H .* sqrt( (S'*W'*V) ./ max(S'*W'*V*(H'*H), options.myeps) );
        else
            H = H .* ( (W'*V) ./ max((W'*W)*H, options.myeps) );
        end
        if options.norm_h ~= 0
            H = normalize_H(H, options.norm_h);
        end
        
        % update S
        if options.orth_w && options.orth_h
            S = S .* sqrt( (W'*V*H') ./ (W'*W*S*(H*H')));
            S = max(S, eps);
        end        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W*S, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        if options.orth_h || options.orth_w
            if options.orth_h && ~options.orth_w
                orth_val = norm(H*H' - eye(rank), 'fro');
            elseif ~options.orth_h && options.orth_w
                orth_val = norm(W'*W - eye(rank), 'fro');
            else 
                orth_val_h = norm(H*H' - eye(rank), 'fro');
                orth_val_w = norm(W'*W - eye(rank), 'fro');
                orth_val = (orth_val_h + orth_val_w) / 2;
            end
            [infos.orth] = [infos.orth orth_val];
        end
        
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('DTPP: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# DTPP: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# DTPP: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end
    
    x.W = W * S;
    x.H = H;

end
