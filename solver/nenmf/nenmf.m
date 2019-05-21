function [x, infos] = nenmf(V, rank, in_options)
% Nesterov's accelerated non-negative matrix factorization (NeNMF).
% The problem of interest is defined as
%
%           min || V - WH ||_F^2 + r(H),
%           where 
%           {V, W, H} > 0 and r(H) is a regularizer. 
%           
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options     
%           type    : 'plain':  NeNMF (min{.5*||V-W*H||_F^2,s.t.,W >= 0 and H >= 0}.)
%
%                   : 'l1r':    L1-norm regularized NeNMF, i.e., r(H) = lambda*||H||_1.
%                   : 'l2r':    L2-norm regularized NeNMF, i.e., r(H) = .5*lambda*||H||_F^2.
%                   : 'mr':     manifold regularized NeNMF, i.e., r(H) = .5*lambda*TR(H*Lp*H^T).
%           lambda  : Rregularization parameter. The default is 1e-3.
%           sim_mat : Similarity matrix constructed by 'constructW'.
%           stop_rule: stop rule for inner loop
%                       - '1' for Projected gradient norm (Default)
%                       - '2' for Normalized projected gradient norm
%                       - '3' for Normalized KKT residual
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
% References
%           N. Guan, D. Tao, Z. Luo, and B. Yuan, 
%           "NeNMF: An Optimal Gradient Method for Non-negative Matrix Factorization",
%           IEEE Transactions on Signal Processing, Vol. 60, No. 6, pp. 2882-2898, Jun. 2012.
%   
%
% Originally written by Naiyang Guan (ny.guan@gmail.com)
% Copyright 2010-2012 by Naiyang Guan and Dacheng Tao
% Modified at Sept. 25 2011
% Modified at Nov. 4 2012
%
% Modified by H.Kasai for NMFLibrar7 on May 21, 2019
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
    
    % set local options 
    local_options = [];
    local_options.type = 'plain';
    local_options.max_inner_iter = 1000;
    local_options.min_inner_iter = 10;
    local_options.lambda = 1e-3;
    local_options.delta_tol = 1e-5;
    local_options.sim_mat = [];
    local_options.stop_rule = 1;     
                        % '1' for Projected gradient norm (Default)
                        % '2' for Normalized projected gradient norm
                        % '3' for Normalized KKT residual
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    if ~strcmp(options.type, 'plain') && ~strcmp(options.type, 'l1r') && ~strcmp(options.type, 'l2r') && ~strcmp(options.type, 'mr') 
        fprintf('Invalid algorithm type: %s. Therfore, we use plain type.\n', options.type);
        options.type = 'plain';
    end    

    if options.verbose > 0
        fprintf('# NeNMF (%s): started ...\n', options.type);           
    end  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;  
    R = zeros(m, n);
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    HVt = H*V'; 
    HHt = H*H';
    WtV = W'*V; 
    WtW = W'*W;
    GradW = W*HHt - HVt';

    switch (options.type)
        case 'plain'
            GradH = WtW*H - WtV;
        case 'l1r'
            GradH = WtW*H - WtV + options.lambda;
        case 'l2r'
            GradH = WtW*H - WtV + options.lambda*H;
        case 'mr'

            if isempty(options.sim_mat)
                error('Similarity matrix must be collected for NeNMF-mr.\n');
            end

            D = diag(sum(options.sim_mat)); 
            Lp = D - options.sim_mat;
            if ~issparse(Lp)
                LpC = norm(Lp);   % Lipschitz constant of mr term
            else
                LpC = norm(full(Lp));
            end
            GradH = WtW*H - WtV + options.lambda*H*Lp;
        otherwise
            error('No such algorithm (%s).\n',options.type);
    end

    init_delta = GetStopCriterion(options.stop_rule, [W', H], [GradW', GradH]);
    tolH = max(options.delta_tol,1e-3)*init_delta;
    tolW = tolH;               % Stopping tolerance
    constV = sum(sum(V.^2));
    delta = init_delta;    

    % select disp_freq 
    disp_freq = set_disp_frequency(options);        
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('NeNMF (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.type, f_val, optgap); 
    end  
    
    % transpose W
    W=W';
    
    % set start time
    start_time = tic();
    prev_time = start_time;

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) && (delta > options.delta_tol*init_delta)

        % Update H with W fixed
        switch (options.type)
            case 'plain'
                [H, iterH] = NNLS(H, WtW, WtV, tolH, options);
            case 'l1r'
                [H, iterH] = NNLS_L1(H, WtW, WtV, tolH, options);
            case 'l2r'
                [H, iterH] = NNLS_L2(H, WtW, WtV, tolH, options);
            case 'mr'
                [H, iterH] = NNLS_MR(H, WtW, WtV, Lp, LpC, tolH, options);
            otherwise
                error('No such algorithm (%s).\n', options.type);
        end

        if iterH <= options.min_inner_iter
            tolH = tolH/10;
        end

        HHt = H*H';   
        HVt = H*V';

        % Update W with H fixed

        [W,iterW,GradW] = NNLS(W, HHt, HVt, tolW, options);
        if iterW <= options.min_inner_iter
            tolW = tolW/10;
        end

        WtW = W*W'; 
        WtV = W*V;
        switch (options.type)
            case 'plain'
                GradH = WtW*H - WtV;
            case 'l1r'
                GradH = WtW*H - WtV + options.lambda;
            case 'l2r'
                GradH = WtW*H - WtV + options.lambda*H;
            case 'mr'
                GradH = WtW*H - WtV + options.lambda*H*Lp;
            otherwise
                error('No such algorithm (%s).\n',options.type);
        end

        delta = GetStopCriterion(options.stop_rule,[W,H],[GradW,GradH]);

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W', H, R, options, infos, epoch, grad_calc_count, elapsed_time);       
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                switch (options.type)
                    case 'plain'
                        objf = sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H));
                    case 'l1r'
                        objf = sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H))+2*options.lambda*sum(sum(H));
                    case 'l2r'
                        objf = sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H))+options.lambda*sum(sum(H.^2));
                    case 'mr'
                        objf = sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H))+options.lambda*sum(sum((H'*H).*Lp));
                    otherwise
                        error('No such algorithm (%s).\n',options.type);
                end

                fprintf('NeNMF (%s): Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e, objf = %.4e, stop cri. = %e\n', ...
                        options.type, epoch, f_val, optgap, elapsed_time - prev_time, objf+constV, delta/init_delta);
            end
        end    

        prev_time = elapsed_time;
    end
    W=W';
    
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# NeNMF (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.type, f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# NeNMF (%s): Max epoch reached (%g).\n', options.type, options.max_epoch);
        elseif delta <= options.delta_tol*init_delta
            fprintf('# NeNMF (%s): Delta tolerance reached: %.4e (delta_tol * init_dela = %.4e).\n', delta, options.delta_tol*init_delta);
        end 
    end    


    x.W = W;
    x.H = H;
end


% Non-negative Least Squares with Nesterov's Optimal Gradient Method
function [H,iter,Grad] = NNLS(Z, WtW, WtV, tol, options)

    if ~issparse(WtW)
        L = norm(WtW);	% Lipschitz constant
    else
        L = norm(full(WtW));
    end
    H = Z;    % Initialization
    Grad = WtW*Z - WtV;     % Gradient
    alpha1 = 1;

    for iter = 1:options.max_inner_iter
        H0 = H;
        H = max(Z-Grad/L,0);    % Calculate sequence 'Y'
        alpha2 = 0.5*(1+sqrt(1+4*alpha1^2));
        Z = H + ((alpha1-1)/alpha2)*(H-H0);
        alpha1 = alpha2;
        Grad = WtW*Z-WtV;

        % Stopping criteria
        if iter >= options.min_inner_iter
            % Lin's stopping condition
            pgn = GetStopCriterion(options.stop_rule, Z, Grad);
            if pgn <= tol
                break;
            end
        end
    end

    Grad = WtW*H - WtV;

end


% L1-norm Regularized Non-negative Least Squares with Nesterov's Optimal Gradient Method
function [H,iter,Grad]= NNLS_L1 (Z, WtW, WtV, tol, options)

    if ~issparse(WtW)
        L = norm(WtW);	% Lipschitz constant
    else
        L = norm(full(WtW));
    end
    H = Z;    % Initialization
    Grad = WtW*Z - WtV + options.lambda;     % Gradient
    alpha1 = 1;

    for iter = 1:options.max_inner_iter
        H0 = H;
        H = max(Z-Grad/L, 0);    % Calculate sequence 'Y'
        alpha2 = 0.5*(1+sqrt(1+4*alpha1^2));
        Z = H + ((alpha1-1)/alpha2)*(H-H0);
        alpha1 = alpha2;
        Grad = WtW*Z - WtV + options.lambda;

        % Stopping criteria
        if iter >= options.min_inner_iter
            pgn = GetStopCriterion(options.stop_rule, Z, Grad);
            if pgn <= tol
                break;
            end
        end
    end

    Grad = WtW*H - WtV + options.lambda;

end


% L2-norm Regularized Non-negative Least Squares with Nesterov's Optimal Gradient Method
function [H,iter,Grad] = NNLS_L2(Z, WtW, WtV, tol, options)

    if ~issparse(WtW)
        L = norm(WtW)+options.lambda;	% Lipschitz constant
    else
        L = norm(full(WtW))+options.lambda;
    end
    H = Z;    % Initialization
    Grad = WtW*Z-WtV+options.lambda*Z;     % Gradient
    alpha1 = 1;

    for iter = 1:options.max_inner_iter
        H0 = H;
        H = max(Z-Grad/L,0);    % Calculate sequence 'Y'
        alpha2 = 0.5*(1+sqrt(1+4*alpha1^2));
        Z = H + ((alpha1-1)/alpha2)*(H-H0);
        alpha1 = alpha2;
        Grad = WtW*Z - WtV + options.lambda*Z;

        % Stopping criteria
        if iter >= options.min_inner_iter
            pgn = GetStopCriterion(options.stop_rule, Z, Grad);
            if pgn <= tol
                break;
            end
        end
    end

    Grad = WtW*H - WtV + options.lambda*H;

end

% Manifold Regularized Non-negative Least Squares with Nesterov's Optimal Gradient Method
function [H,iter,Grad] = NNLS_MR(Z, WtW, WtV, Lp, LpC, tol, options)

    if ~issparse(WtW)
        L = norm(WtW)+options.lambda*LpC;	% Lipschitz constant
    else
        L = norm(full(WtW))+options.lambda*LpC;
    end
    H = Z;    % Initialization
    Grad=WtW*Z-WtV+options.lambda*Z*Lp;     % Gradient
    alpha1=1;

    for iter = 1:options.max_inner_iter
        H0 = H;
        H = max(Z-Grad/L,0);    % Calculate sequence 'Y'
        alpha2 = 0.5*(1+sqrt(1+4*alpha1^2));
        Z = H+((alpha1-1)/alpha2)*(H-H0);
        alpha1 = alpha2;
        Grad = WtW*Z - WtV + options.lambda*Z*Lp;

        % Stopping criteria
        if iter >= options.min_inner_iter
            pgn = GetStopCriterion(options.stop_rule, Z, Grad);
            if pgn <= tol
                break;
            end
        end
    end

    Grad = WtW*H - WtV + options.lambda*H*Lp;

end

function retVal = GetStopCriterion(stop_rule, X, gradX)
    % Stopping Criterions
    % Written by Naiyang (ny.guan@gmail.com)

    switch stop_rule
        case 1
            pGrad = gradX(gradX<0|X>0);
            retVal = norm(pGrad);
        case 2
            pGrad = gradX(gradX<0|X>0);
            pGradNorm = norm(pGrad);
            retVal = pGradNorm/length(pGrad);
        case 3
            resmat = min(X,gradX); 
            resvec = resmat(:);
            deltao = norm(resvec,1);  %L1-norm
            num_notconv = length(find(abs(resvec)>0));
            retVal = deltao/num_notconv;
    end

end

