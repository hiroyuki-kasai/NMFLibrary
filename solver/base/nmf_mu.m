function [x, infos] = nmf_mu(V, rank, in_options)
% Multiplicative upates (MU) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2,
%           where 
%           {V, W, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
%           alg     : mu: Multiplicative upates (MU)
%                       Reference:
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
% Created by H.Kasai on Feb. 16, 2017
% Modified by H.Kasai on Oct. 27, 2017


    % set dimensions and samples
    m = size(V, 1);
    n = size(V, 2);

    % set local options 
    local_options.alg   = 'mu';
    local_options.alpha = 2;
    local_options.delta = 0.1;
    
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
    
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0; 
    
    if ~isfield(options, 'x_init')
        W = rand(m, rank);
        H = rand(rank, n);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end   
    
    if strcmp(options.alg, 'mu_mod')
        delta = eps;
    elseif strcmp(options.alg, 'mu_acc') 
        K = m*n;        
        rhoW = 1+(K+n*rank)/(m*(rank+1)); 
        rhoH = 1+(K+m*rank)/(n*(rank+1));         
    end
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
   
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('MU (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        if strcmp(options.alg, 'mu')
            % update H
            H = H .* (W' * V) ./ (W' * W * H);
            H = H + (H<eps) .* eps;
            
            % update W
            W = W .* (V * H') ./ (W * (H * H'));
            W = W + (W<eps) .* eps;
            
            
            % KL case (To Do)
            %H = H .* (W'*(V./(W*H + 1e-9)))./(sum(W)'*ones(1,samples)); 
            %W = W .* ((V./(W*H + 1e-9))*H')./(ones(vdim,1)*sum(H'));
            
        elseif strcmp(options.alg, 'mu_mod')
            % update H
            WtW = W' * W;
            WtV = W' * V;
            gradH = WtW * H - WtV;
            Hb = max(H, (gradH < 0)* eps);
            H = H - Hb ./ (WtW * Hb + delta) .* gradH;
            
            % update W
            HHt = H * H';
            VHt = V * H';
            gradW = W * HHt - VHt;
            Wb = max(W, (gradW < 0)* eps);
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
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('MU (%s): Epoch = %04d, cost = %.16e, optgap = %.4e\n', options.alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# MU (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg, f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# MU (%s): Max epoch reached (%g).\n', options.alg, options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;

end
