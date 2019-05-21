function [x, infos] = pnmf_vb(V, rank, in_options)
% Probabilistic non-negative matrix factorization (NMF) with Variational Bayesian (VB).
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
%           ard: a boolean indicating whether we use ARD in this model or not.
%           hyperparameters
%               alphatau, betatau: non-negative reals defining prior over noise parameter tau.
%               alpha0, beta0: if using the ARD, non-negative reals defining prior over ARD lambda.
%               lambdaU, lambdaV: if not using the ARD, nonnegative reals defining prior over U and V
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
% The random variables are initialised as follows:
%       (lambdak) alphak_s, betak_s - set to alpha0, beta0
%       (U,V) muU, muV - expectation ('exp') or random ('random')
%       (U,V) tauU, tauV - set to 1
%       (tau) alpha_s, beta_s - using updates
%
%
% References
%       T. Brouwer, J. Frellsen. P. Lio, 
%       "Comparative Study of Inference Methods for Bayesian Nonnegative Matrixm" 
%       ECML PKDD, 2017.
%
%
% Created by H.Kasai on May 21, 2019
% The original 'pyhon' code by the authors is provided as
% https://github.com/ThomasBrouwer/BNMTF_ARD. 
% This code is ported into MATLAB code by H.Kasai.
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
    
    % set local options 
    local_options = [];
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);
    options = mergeOptions(options, in_options);    

    if options.verbose > 0
        fprintf('# Probabilistic NMF with VB started ...\n');           
    end  
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    R_zero = zeros(m, n);

    size_Omega = m * n;
    
    G = gamma_dist();
    T = tn_vector();
    
    if options.ard
        % do nothing for alpha0 and beta0
        options.hyperparams.lambdaU = 0;
        options.hyperparams.lambdaV = 0;
        
        alphak_s    = zeros(rank, 1);
        betak_s     = zeros(rank, 1); 
        exp_lambdak = zeros(rank, 1);       
        exp_loglambdak  = zeros(rank, 1);      
        
        for k = 1 : rank
            alphak_s(k) = options.hyperparams.alpha0;
            betak_s(k) = options.hyperparams.alpha0;
            [exp_lambdak, exp_loglambdak] = update_exp_lambdak(exp_lambdak, exp_loglambdak, alphak_s, betak_s, G, k);
        end  
        
                
    else
        options.hyperparams.alpha0  = 0;
        options.hyperparams.beta0   = 0;
        options.hyperparams.lambdaU = options.hyperparams.lambdaU * ones(m, rank);
        options.hyperparams.lambdaV = options.hyperparams.lambdaV * ones(n, rank);        

        alphak_s = [];
        betak_s = [];
        exp_lambdak = [];        
        exp_loglambdak = [];
    end
    
    %  initialize factors
    mu_U    = zeros(m, rank);
    tau_U   = zeros(m, rank);
    exp_U   = zeros(m, rank);
    var_U   = zeros(m, rank);
    mu_V    = zeros(n, rank);
    tau_V   = zeros(n, rank);  
    exp_V   = zeros(n, rank);
    var_V   = zeros(n, rank);    
    
    init_options = options;
    init_options.exp_U  = exp_U;
    init_options.exp_V  = exp_V;
    init_options.var_U  = var_U;
    init_options.var_V  = var_V;      
    init_options.mu_U   = mu_U;
    init_options.mu_V   = mu_V;       
    init_options.tau_U  = tau_U;
    init_options.tau_V  = tau_V;         
    init_options.hyperparams = options.hyperparams;
    init_options.exp_lambdak = exp_lambdak;
    init_options.ard = options.ard;
    [init_factors, init_factors_opts] = generate_init_factors(V, rank, init_options);    

    exp_U = init_factors.W;
    exp_V = (init_factors.H)';
    var_U = init_factors_opts.var_W;
    var_V = (init_factors_opts.var_H)';
    mu_U = init_factors_opts.mu_W;
    mu_V = (init_factors_opts.mu_H)';
    tau_U = init_factors_opts.tau_W;
    tau_V = (init_factors_opts.tau_H)';        
            

    % Parameter updates tau. '''
    [alpha_s, beta_s] = update_tau(options, V, exp_U, exp_V, var_U, var_V, size_Omega);
    [exp_tau, exp_logtau] = update_exp_tau(alpha_s, beta_s, G);
    

    % select disp_freq 
    disp_freq = set_disp_frequency(options);        
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, exp_U, exp_V', R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        if options.ard
            fprintf('PNMF_VB (ARD): Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
        else
            fprintf('PNMF_VB: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
        end
         
    end  
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           

        if options.ard
            
            for k = 1 : rank
                [alphak_s, betak_s] = update_lambdak(alphak_s, betak_s, exp_U, exp_V, k, options);
                [exp_lambdak, exp_loglambdak] = update_exp_lambdak(exp_lambdak, exp_loglambdak, alphak_s, betak_s, G, k);
            end             
            
        end
        
        % Update U
        for k = 1 : rank
            % update U
            [tau_U, mu_U] = update_U(mu_U, tau_U, V, exp_U, exp_V, var_V, exp_tau, exp_lambdak, k, options);
            % update_exp_U
            [exp_U, var_U] = update_exp(T, exp_U, var_U, mu_U, tau_U, k);
        end    
        
        % Update V
        for k = 1 : rank
            % update V
            [tau_V, mu_V] = update_V(mu_V, tau_V, V, exp_V, exp_U, var_U, exp_tau, exp_lambdak, k, options);
            % update_exp_V
            [exp_V, var_V] = update_exp(T, exp_V, var_V, mu_V, tau_V, k);
        end  
        
        [alpha_s, beta_s] = update_tau(options, V, exp_U, exp_V, var_U, var_V, size_Omega);
        [exp_tau, exp_logtau] = update_exp_tau(alpha_s, beta_s, G);        
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, exp_U, exp_V', R_zero, options, infos, epoch, grad_calc_count, elapsed_time);       
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                if options.ard
                    fprintf('PNMF_VB (ARD): Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
                else
                    fprintf('PNMF_VB: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
                end
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# PNMF_VB (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.ard, f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# PNMF_VB (%s): Max epoch reached (%g).\n', options.ard, options.max_epoch);
        end 
    end
    
    x.W = exp_U;
    x.H = exp_V';

end




function [tau_U, mu_U] = update_U(mu_U, tau_U, V, exp_U, exp_V, var_V, exp_tau, exp_lambdak, k, options)

    m = size(V, 1);

    if options.ard
        lamb = exp_lambdak(k);
    else
        lamb = options.hyperparams.lambdaU(:,k);
    end

    % update tau_U
    tmp = var_V(:,k) + exp_V(:,k).^2;
    tmp2 = repmat(tmp', [m,1]);
    tau_U(:,k) = exp_tau * sum(tmp2, 2);

    % update mu_U
    tmp = V - exp_U * exp_V' + exp_U(:,k) * exp_V(:,k)';
    tmp2 = repmat(exp_V(:,k)', [m,1]);
    mu_U(:,k) = 1 ./ tau_U(:,k) .* (-lamb + exp_tau * sum(tmp .* tmp2, 2));

end

function [tau_V, mu_V] = update_V(mu_V, tau_V, V, exp_V, exp_U, var_U, exp_tau, exp_lambdak, k, options)

    n = size(V, 2);

    if options.ard
        lamb = exp_lambdak(k);
    else
        lamb = options.hyperparams.lambdaV(:,k);
    end

    % update tau_V
    tmp = var_U(:,k) + exp_U(:,k).^2;
    tmp2 = repmat(tmp', [n,1]);
    tau_V(:,k) = exp_tau * sum(tmp2, 2);

    % update mu_V
    tmp = V - exp_U * exp_V' + exp_U(:,k) * exp_V(:,k)';
    tmp2 = repmat(exp_U(:,k)', [n,1]);
    mu_V(:,k) = 1 ./ tau_V(:,k) .* (-lamb + exp_tau * sum(tmp' .* tmp2, 2));

end



function [exp_A, var_A] = update_exp(T, exp_A, var_A, mu_A, tau_A, k)

    exp_A(:,k) = T.expectation(mu_A(:,k), tau_A(:,k));
    var_A(:,k) = T.variance(mu_A(:,k), tau_A(:,k));
    
end

function [alpha_s, beta_s] = update_tau(options, V, exp_U, exp_V, var_U, var_V, size_Omega)
    % Parameter updates tau.
    
    alpha_s = options.hyperparams.alphatau + size_Omega/2.0;
    beta_s = options.hyperparams.betatau + 0.5*exp_square_diff(V, exp_U, exp_V, var_U, var_V);    
    
end



function [exp_tau, exp_logtau] = update_exp_tau(alpha_s, beta_s, gamma)
    % Update expectation tau.
    
    exp_tau = gamma.gamma_expectation(alpha_s, beta_s);
    exp_logtau = gamma.gamma_expectation_log(alpha_s, beta_s);

end


function [alphak_s, betak_s] = update_lambdak(alphak_s, betak_s, exp_U, exp_V, k, options)
    % ''' Parameter updates lambdak.
    
    alphak_s(k) = options.hyperparams.alpha0 + size(exp_U, 1) + size(exp_V, 1);
    betak_s(k) = options.hyperparams.beta0 + sum(exp_U(:,k)) + sum(exp_V(:,k));
    
end
        
        

function [exp_lambdak, exp_loglambdak] = update_exp_lambdak(exp_lambdak, exp_loglambdak, alphak_s, betak_s, gamma, k)
    % Update expectation lambdak.

    exp_lambdak(k) = gamma.gamma_expectation(alphak_s(k), betak_s(k));
    exp_loglambdak(k) = gamma.gamma_expectation_log(alphak_s(k), betak_s(k));

end


function [diff] = exp_square_diff(V, exp_U, exp_V, var_U, var_V)

    % Compute sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]. '''
    tmp =   (V - exp_U * exp_V').^2 ...
            + (var_U+exp_U.^2) * (var_V+exp_V.^2)' ...
            - (exp_U.^2 * (exp_V.^2)');
               
    diff =  sum(sum(tmp));
                          
end
   


