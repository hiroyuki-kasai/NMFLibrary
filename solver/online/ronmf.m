function [x, infos] = ronmf(V, rank, options)
% Robust online non-negative matrix factorization (NMF) with outliers (RONMF) algorithm.
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
%       R. Zhao and Y. F. Tan,
%       "Online nonnegative matrix factorization with outliers,"
%       ICASSP, 2016.
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
    
    if ~isfield(options, 'batch_size')
        batch_size = 1;
    else
        batch_size = options.batch_size;
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
    
    if ~isfield(options, 'x_init')
        Wt = rand(m, rank);
        H = rand(rank, n);
        R = rand(m, n);
    else
        Wt = options.x_init.W;
        H = options.x_init.H;
        R = options.x_init.R;
    end     

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end      

    % initialize
    epoch = 0;
    l = zeros(m, batch_size) + lambda;
    grad_calc_count = 0;  
    
    if verbose > 0
        fprintf('# R-ONMF: started ...\n');           
    end     
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, Wt, H, R);
    infos.cost = f_val;
    optgap = f_val - f_opt;
    infos.optgap = optgap; 
    if verbose > 1
        fprintf('R-ONMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap);    
    end
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
         
    % set start time
    start_time = tic();
    prev_time = start_time;        
    
    % main outer loop
    while (optgap > tol_optgap) && (epoch < max_epoch)      

        % Reset sufficient statistic        
        At = zeros(m, rank);
        Bt = zeros(rank, rank);        
        Ct = zeros(m, rank);

        % main innerr loop
        for t = 1 : batch_size : n - 1

            % Retrieve vt, ht and rt
            vt = V(:, t:t+batch_size-1);
            ht = H(:, t:t+batch_size-1);
            rt = R(:, t:t+batch_size-1);

            % update ht/rt
            ht = ht .* (Wt.' * vt) ./ (Wt.' * (Wt * ht + rt));
            ht = ht + (ht<eps) .* eps;      
            rt = rt .* vt ./ (Wt * ht + rt + l);

            % update sufficient statistics
            At = At + vt *  ht';
            Bt = Bt + ht *  ht'; 
            Ct = Ct + rt *  ht';

            % Update Wt
            Wt = Wt .* At ./ (Wt * Bt + Ct); 
            Wt = Wt + (Wt<eps) .* eps;

            % Update H
            H(:,t:t+batch_size-1) = ht;    
            
            % Update R
            R(:,t:t+batch_size-1) = rt;
            
            grad_calc_count = grad_calc_count + m * batch_size;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % measure cost ad optgap
        f_val = nmf_cost(V, Wt, H, R);
        optgap = f_val - f_opt;   
        
        % update epoch
        epoch = epoch + 1;         

        % store
        infos.epoch = [infos.epoch epoch];
        infos.cost = [infos.cost f_val];
        infos.optgap = [infos.optgap optgap];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % display infos
        if verbose > 1
            if ~mod(epoch, disp_freq)               
                fprintf('R-ONMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);  
            end
        end   
        
        prev_time = elapsed_time;          
    end
    
    if verbose > 0
        if optgap < tol_optgap
            fprintf('# R-ONMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, tol_optgap);
        elseif epoch == max_epoch
            fprintf('# R-ONMF: Max epoch reached (%g).\n', max_epoch);
        end     
    end  
    
    x.W = Wt;
    x.H = H;
    x.R = R;    
end





