function [x, infos] = rnmf(V, rank, options)
% Robust non-negative matrix factorization (NMF) with outliers (RNMF) algorithm.
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Naiyang Guan, Dacheng Tao, Zhigang Luo, and Bo Yuan,
%       "Online nonnegative matrix factorization with robust stochastic approximation,"
%       IEEE Trans. Newral Netw. Learn. Syst., 2012.
%    
%
% Created by H.Sakai and H.Kasai on Feb. 12, 2017

    m = size(V, 1);
    n = size(V, 2); 

    if ~isfield(options, 'lambda')
        lambda = 1;
    else
        lambda = options.lambda;
    end 
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end    
    
    if ~isfield(options, 'x_init')
        W = rand(m, rank);
        H = rand(rank, n);
        R = rand(m, n);
    else
        W = options.x_init.W;
        H = options.x_init.H;
        R = options.x_init.R;
    end     

     
    % initialize
    L = zeros(m, n) + lambda; 
    grad_calc_count = 0;
    
    if verbose > 0
        fprintf('# R-NMF: started ...\n');           
    end      
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, W, H, R);
    infos.cost = f_val;
    if verbose > 1
        fprintf('R-NMF: Epoch = 0000, cost = %.16e\n', f_val); 
    end
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
    infos.optgap = f_val - f_opt;
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end     
         
    % set start time
    start_time = tic();

    % main loop
    for epoch = 1 : max_epoch 
        
        % update H/R/W
        H = H .* (W.' * V) ./ (W.' * (W * H + R));
        R = R .* V./ (W * H + R + L);
        W = W .* (V * H.') ./ ((W * H + R) * H.');
        
        grad_calc_count = grad_calc_count + m*n;

        % measure cost
        f_val = nmf_cost(V, W, H, R);
        % measure elapsed time
        elapsed_time = toc(start_time);
        % measure optimality gap
        optgap = f_val - f_opt;
        
        % store info
        infos.epoch = [infos.epoch epoch];
        infos.cost = [infos.cost f_val];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        
        % display infos
        if verbose > 1
            if ~mod(epoch, disp_freq)            
                fprintf('R-NMF: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end        
    end
    
    if verbose > 0
        if epoch == max_epoch
            fprintf('# R-NMF: Max epoch reached (%g).\n', max_epoch);
        end          
    end
    
    x.W = W;
    x.H = H;
    x.R = R;      
end


