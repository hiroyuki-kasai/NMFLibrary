function [x, infos] = sc_nmf(V, rank, in_options)
% Nonnegative matrix factorization with sparseness constraints (SC-NMF)
%
% The problem of interest is defined as
%
%       min ,
%       where 
%       {V, W, H} > 0.
%
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
% Reference:
%       Patrik O. Hoyer, 
%       "Non-negative matrix factorization with sparseness constraints," 
%       Journal of Machine Learning Research, vol.5, pp.1457-1469, 2004.
%
%
% This file is part of NMFLibrary.
%
% Created by Patrik Hoyer, 2006 (and modified by Silja Polvi-Huttunen, 
% University of Helsinki, Finland, 2014)
%
% Modified by H.Kasai on Jul. 23, 2018
%
% Change log: 
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Oct. 14, 2020 (Haonan Huang): Corrected bugs.
%
%       Jul. 14, 2022 (Hiroyuki Kasai): Fixed algorithm.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options 
    local_options.sW = [];
    local_options.sH = 0.8;
    local_options.init_alg  = 'NNDSVD';
    local_options.stepsizeW = 1.0;
    local_options.stepsizeH = 1.0;    

    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 

    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    H = H ./ (sqrt(sum(H.^2,2)) * ones(1,n));
    R = init_factors.R;
    
    % initialize
    method_name = 'SC-NMF';      
    epoch = 0;    
    grad_calc_count = 0; 

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     
    
    if ~isempty(options.sW)
        L1a = sqrt(n)-(sqrt(n)-1) * options.sW;
        for i = 1 : rank
            W(:,i) = projfunc(W(:,i), L1a, 1, 1); 
        end
    end
    if ~isempty(options.sH) 
        L1s = sqrt(n)-(sqrt(n)-1)*options.sH;
        for i = 1 : rank
            H(i,:) = (projfunc(H(i,:)', L1s, 1, 1))'; 
        end
    end    
    
    stepsizeW = options.stepsizeW; 
    stepsizeH = options.stepsizeH;     
      
   
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    % store additionally different cost
    reg_val = options.lambda * sum(sum(H));
    f_val_total = f_val + reg_val;
    infos.cost_reg = reg_val;
    infos.cost_total = f_val_total;       
    
    if options.verbose > 1
        fprintf('SC-NMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
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

        % update H
        if isempty(options.sH)
            % Update using standard NMF multiplicative update rule
            H = H .* (W' * V) ./ (W' * W * H + 1e-9);

            % Renormalize so rows of H have constant energy
            norms = sqrt(sum(H'.^2));
            H = H ./ (norms' * ones(1, n));  % fixed by Haonan Huang
            W = W .* (ones(m,1) * norms);   % fixed by Haonan Huang
        else
            [H,stepsizeH] = update_with_SC(H,'H',stepsizeH);
        end

        % update W
        if isempty(options.sW)
            % Update using standard NMF multiplicative update rule
            W = W .* (V * H') ./ (W*(H*H') + 1e-9);
            % previous results were produced with this here, instead of W update: 
            %   H = H.*(W'*V)./(W'*W*H + 1e-9);
        else
            [W,stepsizeW] = update_with_SC(W, 'W', stepsizeW);
        end
        
   
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val] = store_nmf_info(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        % store additionally different cost
        reg_val = options.lambda * sum(sum(H));
        f_val_total = f_val + reg_val;
        infos.cost_reg = [infos.cost_reg reg_val];
        infos.cost_total = [infos.cost_total f_val_total];         
        
        % display info
        display_info(method_name, epoch, infos, options);           

    end 

    x.W = W;
    x.H = H;
    
    
    function [Xnew,stepsizeX_new] = update_with_SC(X,name,stepsizeX)
        stepsizeX_new = stepsizeX;
        if strcmp(name,'H')
            dX = W' * (W * H - V);
        else % name is 'W'
            dX = (W * H - V) * H';
        end

        begobj = nmf_cost(V, W, H, R);
        % Make sure to decrease the objective:
        while 1

            Xnew = X - stepsizeX_new*dX; % step to negative gradient, then project
            if strcmp(name, 'H')
                for j=1:rank
                    Xnew(j, :) = (projfunc(Xnew(j, :)', L1s, 1, 1))';
                end
            else
                norms = sqrt(sum(Xnew.^2));
                for j=1:rank 
                    Xnew(:, j) = projfunc(Xnew(:, j), L1a * norms(j), (norms(j)^2), 1); 
                end
            end

            if strcmp(name, 'H')
                newobj = nmf_cost(V, W, Xnew, R);
            else
                newobj = nmf_cost(V, Xnew, H, R);
            end
            if newobj<begobj % objective decreased, we can continue
                break
            end
            stepsizeX_new = stepsizeX_new / 2; % otherwise decrease step size
            if stepsizeX_new < 1e-200 % converged
                return 
            end

        end
        stepsizeX_new = stepsizeX_new*1.2; % slightly increase the step size
    end

end    