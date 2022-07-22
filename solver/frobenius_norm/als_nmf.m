function [x, infos] = als_nmf(V, rank, in_options)
% Alternative least squares (ALS) for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} > 0.
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
% This file is part of NMFLibrary
%
% Created by H.Kasai on Mar. 24, 2017
%
% Change log: 
%
%       Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%       Apr. 22, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jun. 24, 2022 (Hiroyuki Kasai): Added momentum acceleration mode and mofified.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%
    

    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.alg   = 'hals';
    local_options.sub_mode = 'std';
    local_options.alpha = 2;
    local_options.delta = 0.1;
    local_options.inner_max_epoch = 500;
    local_options.inner_max_epoch_parameter = 0.5;       
    local_options.beta0 = 0.5;
    local_options.eta = 1.5; 
    local_options.gammabeta = 1.01;
    local_options.gammabetabar = 1.005; 
    local_options.momentum_h = 0; 
    local_options.momentum_w = 0; 
    local_options.scaling = true;
    local_options.warm_restart = false;    
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);    

    % set paramters
    if ~strcmp(options.alg, 'als') && ~strcmp(options.alg, 'hals') ...
       && ~strcmp(options.alg, 'acc_hals')
        fprintf('Invalid algorithm: %s. Therfore, we use hals (i.e., Hierarchical ALS).\n', options.alg);
        options.alg = 'hals';
    end

    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;  
    
    % initialize
    method_name = sprintf('ALS (%s:%s)', options.alg, options.sub_mode);
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    % intialize for als        
    if strcmp(options.alg, 'acc_hals')
        eit1 = cputime; 
        VHt = V*H'; 
        HHt = H*H'; 
        
        scaling = sum(sum(VHt.*W))/sum(sum( HHt.*(W'*W) )); 
        W = W * scaling;
        
        options_halsupdt = [];
    end  
    
    if options.scaling
        [W, H] = normalize_WH(V, W, H, rank, 'type1');
    end
    
    [options, beta, betamax] = check_momemtum_setting(options);    
    
    if options.warm_restart
        nV = norm(V, 'fro');
        rel_error = zeros(1, options.max_epoch);
        rel_error(1) = sqrt(nV^2 - 2*sum(sum(V * H' .* W)) + sum(sum( H * H' .* (W'*W)))) / nV;          
    end
    W_prev = W; 
    H_prev = H; 
     
     
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
       

        %% update H
        VtW = V'*W;
        WtW = W'*W;
        WtV = W' * V;         
        
        if strcmp(options.alg, 'als')
            
            %H = (W*pinv(W'*W))' * V;
            H = WtW \ WtV;        % H = inv(W'*W) * W' * V;
            H = H .* (H>0);

        elseif strcmp(options.alg, 'hals')
            
            for k=1:rank
                tmp = (VtW(:,k)' - (WtW(:,k)' * H) + (WtW(k,k) * H(k,:))) / WtW(k,k);
                tmp(tmp<=eps) = eps;
                H(k,:) = tmp;
            end 
            
        elseif strcmp(options.alg, 'acc_hals')

            eit1 = cputime; 
            options_halsupdt.max_epoch = change_inner_max_epoch(V, W, options);
            H = HALSupdt(H, WtW, WtV, eit1, options.alpha, options.delta, options_halsupdt); 

        end
        
        % perform momentum for H
        if strcmp(options.sub_mode, 'momentum')
            [H, H_tmp1, H_tmp2] = do_momentum_h(H, H_prev, beta, epoch, options);
        end
         
        
        
        %% update W
        VHt = V * H';
        HHt = H * H';
            
        if strcmp(options.alg, 'als')
            
            %W = ((inv(H*H')*H)*V')';
            W = VHt / HHt;        % W = V * H' * inv(H*H');
            W = (W>0) .* W;
            
            % normalize columns to unit 
            W = W ./ (repmat(sum(W), m, 1)+eps); 

        elseif strcmp(options.alg, 'hals')

            for k=1:rank
                tmp = (VHt(:,k) - (W * HHt(:,k)) + (W(:,k) * HHt(k,k))) / HHt(k,k);
                tmp(tmp<=eps) = eps;
                W(:,k) = tmp;
            end
            
        elseif strcmp(options.alg, 'acc_hals')
            
%            if epoch > 0 % Do not recompute A and B at first pass
                % Use actual computational time instead of estimates rhoU
                eit1 = cputime; 
                eit1 = cputime-eit1; 
%           end
            options_halsupdt.max_epoch = change_inner_max_epoch(V', H', options);
            W = HALSupdt(W', HHt',VHt', eit1, options.alpha, options.delta, options_halsupdt); 
            W = W';

        end 
        
        % perform momentum for W 
        if strcmp(options.sub_mode, 'momentum')
            [W, H, W_tmp1] = do_momentum_w(W, W_prev, H, H_prev, H_tmp1, beta, epoch, options);
        end        
        
        % perform warm_restart
        if options.warm_restart
            [W, H, W_prev, H_prev, rel_error, beta, betamax, options] = ...
                warm_restart(V, W, H, rank, W_prev, H_prev, W_tmp1, H_tmp1, H_tmp2, rel_error, beta, betamax, epoch, options);
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