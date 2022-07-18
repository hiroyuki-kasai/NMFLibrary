function [x, infos] = fro_mu_nmf(V, rank, in_options)
% Frobenius-norm based multiplicative upates (MU) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min f(V, W, H),
%       where 
%       {V, W, H} >= 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
%           alg     : mu: Multiplicative upates (MU)
%                       Reference for Euclidean distance and Kullback?Leibler divergence (KL):
%                           Daniel D. Lee and H. Sebastian Seung,
%                           "Algorithms for non-negative matrix factorization,"
%                           NIPS 2000. 
%
%                   : mu_mod: Modified multiplicative upates (MU)
%                       Reference:
%                           C.-J. Lin,
%                           "On the convergence of multiplicative update algorithms for nonnegative matrix factorization,"
%                           IEEE Trans. Neural Netw. vol.18, no.6, pp.1589?1596, 2007. 
%
%                   : mu_acc: Accelerated multiplicative updates (Accelerated MU)
%                       Reference:
%                           N. Gillis and F. Glineur, 
%                           "Accelerated Multiplicative Updates and hierarchical ALS Algorithms for Nonnegative 
%                           Matrix Factorization,", 
%                           Neural Computation 24 (4), pp. 1085-1105, 2012. 
%                           See http://sites.google.com/site/nicolasgillis/.
%                           The corresponding code is originally created by the authors, 
%                           Then, it is modifided by H.Kasai.
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
%
% This file is part of NMFLibrary
%
% Created by H.Kasai on Feb. 16, 2017
%
% Some parts are borrowed after modifications from the codes by Patrik Hoyer, 2006 
% (and modified by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014)
%
% Change log: 
%
%       Jul. 20, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%       Apr. 20, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       May. 27, 2019 (Hiroyuki Kasai): Remove divergence-based algorithms.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.alg       = 'mu';
    local_options.norm_h    = 0;
    local_options.norm_w    = 1;    
    local_options.alpha     = 2;
    local_options.delta     = 0.1;
    local_options.myeps     = 1e-16;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);        
    
    if ~strcmp(options.alg, 'mu') && ~strcmp(options.alg, 'mu_mod') && ~strcmp(options.alg, 'mu_acc') 
        fprintf('Invalid algorithm: %s. Therfore, we use mu (i.e., multiplicative update).\n', options.alg);
        options.alg = 'mu';
    end
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0;
    
    if strcmp(options.alg, 'mu_mod')
        delta = options.myeps;
    elseif strcmp(options.alg, 'mu_acc') 
        K = m*n;        
        rhoW = 1+(K+n*rank)/(m*(rank+1)); 
        rhoH = 1+(K+m*rank)/(n*(rank+1));         
    end
    method_name = sprintf('Fro-MU (%s)', options.alg);

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
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

        if strcmp(options.alg, 'mu')
            
            % update H
            H = H .* (W' * V) ./ (W' * W * H);
            H = H + (H<options.myeps) .* options.myeps;

            % update W
            W = W .* (V * H') ./ (W * (H * H'));
            W = W + (W<options.myeps) .* options.myeps;
                

        elseif strcmp(options.alg, 'mu_mod')
            
            % update H
            WtW = W' * W;
            WtV = W' * V;
            gradH = WtW * H - WtV;
            Hb = max(H, (gradH < 0)* options.myeps);
            H = H - Hb ./ (WtW * Hb + delta) .* gradH;
            
            % update W
            HHt = H * H';
            VHt = V * H';
            gradW = W * HHt - VHt;
            Wb = max(W, (gradW < 0)* options.myeps);
            W = W - Wb ./ (Wb * HHt + delta) .* gradW;
            
            S = sum(W,1);
            W = W ./ repmat(S,m,1);
            H = H .* repmat(S',1,n);
            
        elseif strcmp(options.alg, 'mu_acc')
            
            % update W
            gamma = 1; 
            eps0 = 1; 
            j = 1;
            VHt = V * H';
            HHt = H * H.';            
            while j <= floor(1+rhoW*options.alpha) && gamma >= options.delta*eps0
                W0 = W; 
                %W = max(1e-16, W .* (V * H' ./ (W * (H * H.')))); 
                W = max(1e-16, W .* (VHt ./ (W * HHt)));
                if j == 1
                    eps0 = norm(W0-W, 'fro'); 
                end
                gamma = norm(W0-W, 'fro');  
                j = j+1;
            end
            
            % Update H
            gamma = 1; 
            eps0 = 1; 
            j = 1; 
            WtV = W'*V;
            WtW = W.' * W;            
            while j <= floor(1+rhoH*options.alpha) &&  gamma >= options.delta*eps0
                H0 = H;
                %H = max(1e-16, H .* (W'*V ./ (W.' * W * H))); 
                H = max(1e-16, H .* (WtV ./ (WtW * H))); 
                if j == 1
                    eps0 = norm(H0-H, 'fro'); 
                end
                gamma = norm(H0-H, 'fro');  
                j = j+1;
            end     
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;

end