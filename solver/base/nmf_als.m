function [x, infos] = nmf_als(V, rank, options)
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
%       options     
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
% Created by H.Kasai on Mar. 24, 2017
% Modified by H.Kasai on Apr. 04, 2017


    m = size(V, 1);
    n = size(V, 2); 
    
    if ~isfield(options, 'alg')
        alg = 'hals';
    else
        if ~strcmp(options.alg, 'als') && ~strcmp(options.alg, 'hals') ...
           && ~strcmp(options.alg, 'acc_hals')
            fprintf('Invalid algorithm: %s. Therfore, we use hals (i.e., Hierarchical ALS).\n', options.alg);
            alg = 'hals';
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
    
    if strcmp(alg, 'acc_hals')       
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
        fprintf('ALS (%s): Epoch = 000, cost = %.16e, optgap = %.4e\n', alg, f_val, optgap); 
    end  
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end    
    
    if strcmp(alg, 'acc_hals')
        eit1 = cputime; 
        VHt = V*H'; 
        HHt = H*H'; 
        eit1 = cputime-eit1; 
        
        scaling = sum(sum(VHt.*W))/sum(sum( HHt.*(W'*W) )); 
        W = W*scaling;         
    end

    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)           

        if strcmp(alg, 'als')
            % update H
            %H = (W*pinv(W'*W))' * V;
            H = (W'*W) \ W' * V;        % H = inv(W'*W) * W' * V;
            H = H .* (H>0);
            % update W
            %W = ((inv(H*H')*H)*V')';
            W = V * H' / (H*H');        % W = V * H' * inv(H*H');
            W = (W>0) .* W;
            % normalize columns to unit 
            W = W ./ (repmat(sum(W),m,1)+eps); 

        elseif strcmp(alg, 'hals')
            % update H
            VtW = V'*W;
            WtW = W'*W;            
            for k=1:rank
                tmp = (VtW(:,k)'-(WtW(:,k)'*H)+(WtW(k,k)*H(k,:)))./WtW(k,k);
                tmp(tmp<=eps) = eps;
                H(k,:) = tmp;
            end 
            
            % update W
            VHt = V*H';
            HHt = H*H';
            for k=1:rank
                tmp = (VHt(:,k)-(W * HHt(:,k))+(W(:,k)*HHt(k,k))) ./ HHt(k,k);
                tmp(tmp<=eps) = eps;
                W(:,k) = tmp;
            end
            
        elseif strcmp(alg, 'acc_hals')
            
            % Update of W
            if epoch > 0, % Do not recompute A and B at first pass
                % Use actual computational time instead of estimates rhoU
                eit1 = cputime; 
                VHt = V*H'; 
                HHt = H*H'; 
                eit1 = cputime-eit1; 
            end
            W = HALSupdt(W',HHt',VHt', eit1, alpha, delta); 
            W = W';
            
            % Update of H
            eit1 = cputime; 
            WtV = W'*V; 
            WtW = W'*W; 
            eit1 = cputime-eit1;
            H = HALSupdt(H, WtW, WtV, eit1, alpha, delta); 

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
                fprintf('ALS (%s): Epoch = %03d, cost = %.16e, optgap = %.4e\n', alg, epoch, f_val, optgap);
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


% Code for updating step for accelerated HALS NMF
% Originally created by N. Gillis
%   See https://sites.google.com/site/nicolasgillis/publications.
%
% Update of V <- HALS(M,U,V)
% i.e., optimizing min_{V >= 0} ||M-UV||_F^2 
% with an exact block-coordinate descent scheme
function V = HALSupdt(V,UtU,UtM,eit1,alpha,delta)
    [r,n] = size(V); 
    eit2 = cputime; % Use actual computational time instead of estimates rhoU
    cnt = 1; % Enter the loop at least once
    eps = 1; eps0 = 1; eit3 = 0;
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
