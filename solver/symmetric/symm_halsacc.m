function [x, infos] = symm_halsacc(V, rank, in_options)
% Symmetric non-negative matrix factorization by HALS (SymHALS).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2 + alpha * ||W-H'||_F^2,
%       where 
%       {V, W, H} >= 0, and W is close to H'.
%
% Given a symmetric non-negative matrix V, factorized non-negative matrices {W, H(close to W')} are calculated.
%
%
% Inputs:
%       V           : (m x m) symmetric non-negative matrix to factorize
%       rank        : rank
%       in_options 
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x m)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
% References
%       Z. Zhu, X. Li, K. Liu, Q. Li, 
%       "Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization",
%       NIPS, 2018.
%
%
% This file is part of NMFLibrary.  
%
% Originally created by Xiao Li et al.
%   See https://github.com/lixiao0982/Dropping-Symmetric-for-Symmetric-NMF.
%       This code is modified from ACC_Hals developed by N. Gillis and F.Glineur. 
%       For this, see http://sites.google.com/site/nicolasgillis/.
%
% Modified by H.Kasai on June 24, 2019 for NMFLibrary
%
% Change log: 
%
%       Jul. 14, 2022 (Hiroyuki Kasai): Fixed algorithm.
%
    

    % set dimensions and samples
    [m] = size(V, 1);
 
    % set local options
    local_options = [];
    local_options.lambda    = 0.1;
    local_options.alpha     = 0;
    local_options.delta     = 0; 
    local_options.init_alg  = 'symm_mean';
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end       
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      

    % initialize
    method_name = 'Symm (HALS)';
    epoch = 0;    
    grad_calc_count = 0;   

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

        % update W
        V1 = [V sqrt(options.lambda)*H']; 
        H1 = [H sqrt(options.lambda)*eye(rank)];
        eit1 = cputime; 
        V1H1t = V1*H1'; 
        H1H1t = H1*H1'; 
        eit1 = cputime-eit1; 
        W = HALSupdt(W',H1H1t', V1H1t', eit1, options.alpha, options.delta); 
        W = W';
        
        % update V
        V2 = [V; sqrt(options.lambda)*W']; 
        W2 = [W; sqrt(options.lambda)*eye(rank)];
        eit1 = cputime; 
        W2tV2 = W2'*V2; 
        W2tW2 = W2'*W2; 
        eit1 = cputime-eit1;
        H = HALSupdt(H, W2tW2, W2tV2, eit1, options.alpha, options.delta); 
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*m;        
        
        % update epoch
        epoch = epoch + 1;         

        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);    

        % display info
        display_info(method_name, epoch, infos, options);        

    end 

    x.W = H;
    x.H = H';

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