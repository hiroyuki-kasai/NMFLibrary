function [x, infos] = nmf_mu(V, rank, in_options)
% Multiplicative upates (MU) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%           min f(V, W, H),
%           where 
%           {V, W, H} >= 0.
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
%                       Reference for Amari alpha divergence:
%                           A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
%                           "Extended SMART algorithms for non-negative matrix factorization,"
%                           Artificial Intelligence and Soft Computing, 2006.
%
%                           min D(V||R) = sum(V(:).^alpha .* R(:).^(1-d_alpha) - d_alpha*V(:) + (d_alpha-1)*R(:)) / (alpha*(d_alpha-1)), 
%                           where R = W*H.
%
%                           - Pearson's distance (d_alpha=2)
%                           - Hellinger's distance (d_alpha=0.5)
%                           - Neyman's chi-square distance (d_alpha=-1)
%
%                       Reference for beta divergence:
%                           A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
%                           "Extended SMART algorithms for non-negative matrix factorization,"
%                           Artificial Intelligence and Soft Computing, 2006.
%
%                           min D(V||W*H)
%
%                                               | sum(V(:).^d_beta + (d_beta-1)*R(:).^d_beta - ...
%                                               |     d_beta*V(:).*R(:).^(d_beta-1)) / ...
%                                               |     (d_beta*(d_beta-1))                    (d_beta \in{0 1}
%                          where D(V||R) =      |                                          
%                                               | sum(V(:).*log(V(:)./R(:)) - V(:) + R(:)) (d_beta=1)
%                                               |
%                                               | sum(V(:)./R(:) - log(V(:)./R(:)) - 1)   (d_beta=0)
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
% Created by H.Kasai on Feb. 16, 2017
%
% Some parts are borrowed after modifications from the codes by Patrik Hoyer, 2006 
% (and modified by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014)
%
% Change log: 
%
%   Jul. 20, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%   Apr. 20, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
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
    local_options.metric    = 'EUC'; % 'EUC' (default) or 'KL'  
    local_options.d_alpha   = -1; % for alpha divergence
    local_options.d_beta    = 0; % for beta divergence 
    local_options.myeps     = 1e-16;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);        
    
    if ~strcmp(options.alg, 'mu') && ~strcmp(options.alg, 'mu_mod') && ~strcmp(options.alg, 'mu_acc') 
        fprintf('Invalid algorithm: %s. Therfore, we use mu (i.e., multiplicative update).\n', options.alg);
        options.alg = 'mu';
    end
    
    if options.verbose > 0
        fprintf('# MU (%s): started ...\n', options.alg);           
    end  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0; 
    
    if strcmp(options.alg, 'mu_mod')
        delta = options.myeps;
    elseif strcmp(options.alg, 'mu_acc') 
        K = m*n;        
        rhoW = 1+(K+n*rank)/(m*(rank+1)); 
        rhoH = 1+(K+m*rank)/(n*(rank+1));         
    end
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
    % store initial info
    clear infos;
    metric_param = [];
    if strcmp(options.metric, 'ALPHA-D')
        metric_param = options.d_alpha;
    elseif strcmp(options.metric, 'BETA-D')
        metric_param = options.d_beta;        
    end
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0, options.metric, metric_param);
    
    if options.verbose > 1
        fprintf('MU (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        if strcmp(options.alg, 'mu')
            if strcmp(options.metric, 'EUC')
                
                % update H
                H = H .* (W' * V) ./ (W' * W * H);
                H = H + (H<options.myeps) .* options.myeps;

                % update W
                W = W .* (V * H') ./ (W * (H * H'));
                W = W + (W<options.myeps) .* options.myeps;
                
            elseif strcmp(options.metric, 'KL')
                
                % update W
                W = W .* ((V./(W*H + options.myeps))*H')./(ones(m,1)*sum(H'));
                if options.norm_w ~= 0
                    W = normalize_W(W, options.norm_w);
                end                
                
                % update H
                H = H .* (W'*(V./(W*H + options.myeps)))./(sum(W)'*ones(1,n));
                if options.norm_h ~= 0
                    H = normalize_H(H, options.norm_h);
                end                    

            elseif strcmp(options.metric, 'ALPHA-D')
                
                % update W
                W = W .* ( ((V+options.myeps) ./ (W*H+options.myeps)).^options.d_alpha * H').^(1/options.d_alpha);
                if options.norm_w ~= 0
                    W = normalize_W(W, options.norm_w);
                end
                W = max(W, options.myeps);

                % update H
                H = H .* ( (W'*((V+options.myeps)./(W*H+options.myeps)).^options.d_alpha) ).^(1/options.d_alpha);
                if options.norm_h ~= 0
                    H = normalize_H(H, options.norm_h);
                end
                H = max(H, options.myeps);
                
            elseif strcmp(options.metric, 'BETA-D')
                
                WH = W * H;
                
                % update W
                W = W .* ( ((WH.^(options.d_beta-2) .* V)*H') ./ max(WH.^(options.d_beta-1)*H', options.myeps) );
                             
                if options.norm_w ~= 0
                    W = normalize_W(W, options.norm_w);
                end
                
                WH = W * H;

                % update H
                H = H .* ( (W'*(WH.^(options.d_beta-2) .* V)) ./ max(W'*WH.^(options.d_beta-1), options.myeps) );
                if options.norm_h ~= 0
                    H = normalize_H(H, options.norm_h);
                end

            end
            
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
            % Update of W
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
            % Update of H
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
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time, options.metric, metric_param);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('MU (%s): Epoch = %04d, cost = %.16e, optgap = %.4e\n', options.alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# MU (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg, f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# MU (%s): Max epoch reached (%g).\n', options.alg, options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;

end
