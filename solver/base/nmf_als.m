function [x, infos] = nmf_als(V, rank, in_options)
% Alternative least squares (ALS) for non-negative matrix factorization (NMF).
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
%           alg     : als: Alternative least squares (ALS)
%
%                   : hals: Hierarchical alternative least squares (Hierarchical ALS)
%                       Reference:
%                           Andrzej Cichocki and PHAN Anh-Huy,
%                           "Fast local algorithms for large scale nonnegative matrix and tensor factorizations,"
%                           IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, 
%                           vol. 92, no. 3, pp. 708-721, 2009.
%
%                   : acc_hals: Accelerated hierarchical alternative least squares (Accelerated HALS)
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
% Outputs:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x n)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
%
% Created by H.Kasai on Mar. 24, 2017
%
% Change log: 
%
%   Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   Apr. 22, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.alg   = 'hals';
    local_options.alpha = 2;
    local_options.delta = 0.1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);    

    % set paramters
    if ~strcmp(options.alg, 'als') && ~strcmp(options.alg, 'hals') ...
       && ~strcmp(options.alg, 'acc_hals')
        fprintf('Invalid algorithm: %s. Therfore, we use hals (i.e., Hierarchical ALS).\n', options.alg);
        options.alg = 'hals';
    end
    
    if options.verbose > 0
        fprintf('# ALS (%s): started ...\n', options.alg);           
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

    if strcmp(options.alg, 'acc_hals')
        eit1 = cputime; 
        VHt = V*H'; 
        HHt = H*H'; 
        eit1 = cputime-eit1; 
        
        scaling = sum(sum(VHt.*W))/sum(sum( HHt.*(W'*W) )); 
        W = W * scaling;         
    end    
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);        
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('ALS (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, f_val, optgap); 
    end  
    
    % set start time
    start_time = tic();
    prev_time = start_time;

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        if strcmp(options.alg, 'als')
            % update H
            %H = (W*pinv(W'*W))' * V;
            H = (W'*W) \ W' * V;        % H = inv(W'*W) * W' * V;
            H = H .* (H>0);
            % update W
            %W = ((inv(H*H')*H)*V')';
            W = V * H' / (H*H');        % W = V * H' * inv(H*H');
            W = (W>0) .* W;
            % normalize columns to unit 
            W = W ./ (repmat(sum(W), m, 1)+eps); 

        elseif strcmp(options.alg, 'hals')
            % update H
            VtW = V'*W;
            WtW = W'*W;            
            for k=1:rank
                tmp = (VtW(:,k)' - (WtW(:,k)' * H) + (WtW(k,k) * H(k,:))) / WtW(k,k);
                tmp(tmp<=eps) = eps;
                H(k,:) = tmp;
            end 
            
            % update W
            VHt = V*H';
            HHt = H*H';
            for k=1:rank
                tmp = (VHt(:,k) - (W * HHt(:,k)) + (W(:,k) * HHt(k,k))) / HHt(k,k);
                tmp(tmp<=eps) = eps;
                W(:,k) = tmp;
            end
            
        elseif strcmp(options.alg, 'acc_hals')
            
            % Update of W
            if epoch > 0 % Do not recompute A and B at first pass
                % Use actual computational time instead of estimates rhoU
                eit1 = cputime; 
                VHt = V*H'; 
                HHt = H*H'; 
                eit1 = cputime-eit1; 
            end
            W = HALSupdt(W',HHt',VHt', eit1, options.alpha, options.delta); 
            W = W';
            
            % Update of H
            eit1 = cputime; 
            WtV = W'*V; 
            WtW = W'*W; 
            eit1 = cputime-eit1;
            H = HALSupdt(H, WtW, WtV, eit1, options.alpha, options.delta); 

        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time);       
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('ALS (%s): Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', options.alg, epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
      
        prev_time = elapsed_time;          
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# ALS (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg, f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# ALS (%s): Max epoch reached (%g).\n', options.alg, options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;

end


% Code for updating step for accelerated HALS NMF
% Originally created by N. Gillis
%   See https://sites.google.com/site/nicolasgillis/publications.
%
% Update of V <- HALS(M,U,V)
% i.e., optimizing min_{V >= 0} ||M-UV||_F^2 
% with an exact block-coordinate descent scheme
function V = HALSupdt(V,UtU,UtM,eit1,alpha,delta)
    [r, n] = size(V); 
    eit2 = cputime; % Use actual computational time instead of estimates rhoU
    cnt = 1; % Enter the loop at least once
    eps = 1; 
    eps0 = 1; 
    eit3 = 0;
    while cnt == 1 || (cputime-eit2 < (eit1+eit3)*alpha && eps >= (delta)^2*eps0)
        nodelta = 0; if cnt == 1, eit3 = cputime; end
            for k = 1 : r
                deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
                V(k,:) = V(k,:) + deltaV;
                nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
                if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
            end
        if cnt == 1
            eps0 = nodelta; 
            eit3 = cputime-eit3; 
        end
        eps = nodelta; cnt = 0; 
    end
end
