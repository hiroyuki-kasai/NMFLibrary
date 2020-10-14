function [init_factors, init_factors_opts] = generate_init_factors(V, rank, options)
% Initialization algorithm 
%
% Created by H.Kasai on May 21, 2019
%
% Change log: 
%
% Modified by H.Huangi on Oct. 14, 2020 (Added 'LPinit')

    % V = WH + R

    m = size(V, 1);
    n = size(V, 2); 

    init_factors_opts = [];
    
    
    
    %% generate W and H
    generate_wh_init = true;
    
    if isfield(options, 'x_init') 
        if isfield(options.x_init, 'W') && isfield(options.x_init, 'H')
            init_factors.W = options.x_init.W;
            init_factors.H = options.x_init.H;    
            generate_wh_init = false;
        end
        
        if isfield(options.x_init, 'R')
            init_factors.R = options.x_init.R;
        end
    end
        
    
    if generate_wh_init
        
        if ~isfield(options, 'init_alg')
            alg = 'random';
        else
            alg = options.init_alg;
        end
            
        switch(alg)
            
            case 'LPinit'
                H = LPinitSemiNMF(V, rank);
                W = V * pinv(H);
                
                init_factors.W = W;
                init_factors.H = H;             

            case 'random'

                W = rand(m, rank);
                H = rand(rank, n);

                init_factors.W = W;
                init_factors.H = H;  
                
            case 'ones'
                
                W = ones(m, rank);
                H = ones(rank, n);

                init_factors.W = W;
                init_factors.H = H;      
                
            case 'semi_random' % for SemiNMF
                
                H = rand(rank, n);
                W = V / H; % V * inv(H)

                init_factors.W = W;
                init_factors.H = H; 
                
            case 'symm'
                
                W = ones(m, rank);

                init_factors.W = W;
                init_factors.H = W';  
                
            case 'symm_mean'
                
                % make sure that entries of H fall into the interval [0, 2*sqrt(m/k)],
                % where 'm' is the average of all entries of V.
                % See https://github.com/dakuang/symnmf
                
                W = 2 * full(sqrt(mean(mean(V)) / rank)) * rand(m, rank);

                init_factors.W = W;
                init_factors.H = W';  
                
            case 'NNDSVD'
                
                [W, H] = NNDSVD(abs(V), rank, 0);      

                init_factors.W = W;
                init_factors.H = H;  
                
            case 'kmeans'
                
                [label, center] = litekmeans(V',rank, 'maxIter', 10);
                center = max(0,center);
                W = center';
                WTW = W'*W;
                WTW = max(WTW, WTW');
                WTX = W'*V;
                H = max(0, WTW\WTX);
                
                init_factors.W = W;
                init_factors.H = H;      

            case {'prob_expectation', 'prob_random'}


                T = tn_vector();

                if strcmp(alg, 'prob_random')
                    E = exponential_dist();
                end

                exp_U   = options.exp_U;
                exp_V   = options.exp_V;
                var_U   = options.var_U;
                var_V   = options.var_V;      
                mu_U    = options.mu_U;
                mu_V    = options.mu_V;             
                tau_U   = options.tau_U;
                tau_V   = options.tau_V;               

                for k = 1 : rank
                    for i = 1 : m
                        tau_U(i, k) = 1;

                        if options.ard
                            hyperparam = options.exp_lambdak(k);
                        else
                            hyperparam = options.hyperparams.lambdaU(i, k);
                        end

                        if strcmp(alg, 'prob_random')
                            mu_U(i, k) = E.exponential_draw(hyperparam);              
                        else
                            mu_U(i, k) = 1.0/hyperparam;
                        end
                    end
                end

                for k = 1 : rank
                    for j = 1 : n
                        tau_V(j, k) = 1;

                        if options.ard
                            hyperparam = options.exp_lambdak(k);
                        else
                            hyperparam = options.hyperparams.lambdaV(j, k);
                        end

                        if strcmp(alg, 'prob_random')
                            mu_V(j, k) = E.exponential_draw(hyperparam);              
                        else
                            mu_V(j, k) = 1.0/hyperparam;
                        end
                    end
                end   

                for k = 1 : rank
                    [exp_U, var_U] = update_exp(T, exp_U, var_U, mu_U, tau_U, k);
                end

                for k = 1 : rank
                    [exp_V, var_V] = update_exp(T, exp_V, var_V, mu_V, tau_V, k);
                end


                init_factors.W = exp_U;
                init_factors.H = exp_V';
                init_factors_opts.var_W = var_U;
                init_factors_opts.var_H = var_V';
                init_factors_opts.mu_W  = mu_U;
                init_factors_opts.mu_H  = mu_V';               
                init_factors_opts.tau_W = tau_U;
                init_factors_opts.tau_H = tau_V';            

            otherwise 

                % do random initialization 

                W = rand(m, rank);
                H = rand(rank, n);

                init_factors.W = W;
                init_factors.H = H;                 

        end
        
    end
    
    if isfield(options, 'norm_w')
        if options.norm_w ~= 0
            % normalize W
            init_factors.W = normalize_W(init_factors.W, options.norm_w);
        end
    end

    if isfield(options, 'norm_h')    
        if options.norm_h ~= 0
            % normalize H
            init_factors.H = normalize_H(init_factors.H, options.norm_h);
        end  
    end
    
    
    %% generate R
    generate_r_init = true;
    
    if options.x_init_robust 
        if isfield(options, 'x_init') 
            if isfield(options.x_init, 'R')
                init_factors.R = options.x_init.R;
                generate_r_init = false;
            end
        end

        if generate_r_init
            init_factors.R = rand(m, n);
        end
    else
        init_factors.R = zeros(m, n);
    end
    
end


function [exp_A, var_A] = update_exp(T, exp_A, var_A, mu_A, tau_A, k)

    exp_A(:,k) = T.expectation(mu_A(:,k), tau_A(:,k));
    var_A(:,k) = T.variance(mu_A(:,k), tau_A(:,k));
    
end


