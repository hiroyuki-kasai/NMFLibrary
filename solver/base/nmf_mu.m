function [x, infos] = nmf_mu(V, rank, options)
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
%       options 
%           alg     : mu: Multiplicative upates (MU)
%                       Reference:
%                           Daniel D. Lee and H. Sebastian Seung,
%                           "Algorithms for non-negative matrix factorization,"
%                           NIPS 2000. 
%
%                   : mod_mu: Modified multiplicative upates (MU)
%                       Reference:
%                           C.-J. Lin,
%                           "On the convergence of multiplicative update algorithms for nonnegative matrix factorization,"
%                           IEEE Trans. Neural Netw. vol.18, no.6, pp.1589?1596, 2007. 
%
%                   : acc_mu: Accelerated multiplicative updates (Accelerated MU)
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
% Modified by H.Kasai on Apr. 04, 2017

    m = size(V, 1);
    n = size(V, 2); 
    
    if ~isfield(options, 'alg')
        alg = 'mu';
    else
        if ~strcmp(options.alg, 'mu') && ~strcmp(options.alg, 'mod_mu') && ~strcmp(options.alg, 'als') ...
                && ~strcmp(options.alg, 'acc_mu') 
            fprintf('Invalid algorithm: %s. Therfore, we use mu (i.e., multiplicative update).\n', options.alg);
            alg = 'mu';
        else
            alg = options.alg;
        end
    end     

    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end   
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end       

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end

    if ~isfield(options, 'x_init')
        W = rand(m, rank);
        H = rand(rank, n);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end 
    
    %if strcmp(options.alg, 'acc_mu') || strcmp(options.alg, 'acc_mu_new')
    if strcmp(alg, 'acc_mu')       
        if ~isfield(options, 'alpha')
            alpha = 2;
        else
            alpha = options.alpha;
        end 
        
        if ~isfield(options, 'delta')
            delta = 0.1;
        else
            delta = options.delta;
        end          
    end
    

    % initialize
    epoch = 0;    
    R = zeros(m, n);
    grad_calc_count = 0; 
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, W, H, R);
    infos.cost = f_val;
    optgap = f_val - f_opt;
    infos.optgap = optgap;   
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
    if verbose > 0
        fprintf('MU (%s): Epoch = 000, cost = %.16e, optgap = %.4e\n', alg, f_val, optgap); 
    end  
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end    

    if strcmp(alg, 'mod_mu')
        delta = eps;
    elseif strcmp(alg, 'acc_mu') 
        K = m*n;        
        rhoW = 1+(K+n*rank)/(m*(rank+1)); 
        rhoH = 1+(K+m*rank)/(n*(rank+1));         
    end
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)           

        if strcmp(alg, 'mu')
            % update H
            H = H .* (W' * V) ./ (W' * W * H);
            H = H + (H<eps) .* eps;
            
            % update W
            W = W .* (V * H') ./ (W * (H * H'));
            W = W + (W<eps) .* eps;
            
        elseif strcmp(alg, 'mod_mu')
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
            
        elseif strcmp(alg, 'acc_mu')
            % Update of W
            gamma = 1; 
            eps0 = 1; 
            j = 1;
            VHt = V * H';
            HHt = H * H.';            
            while j <= floor(1+rhoW*alpha) && gamma >= delta*eps0
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
            while j <= floor(1+rhoH*alpha) &&  gamma >= delta*eps0
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

        % calculate cost and optgap 
        f_val = nmf_cost(V, W, H, R);
        optgap = f_val - f_opt;    
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos.epoch = [infos.epoch epoch];
        infos.cost = [infos.cost f_val];
        infos.optgap = [infos.optgap optgap];     
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % display infos
        if verbose > 0
            if ~mod(epoch, disp_freq)
                fprintf('MU (%s): Epoch = %03d, cost = %.16e, optgap = %.4e\n', alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', max_epoch);
    end 
    
    x.W = W;
    x.H = H;

end
