function [x, infos] = onmf_acc(V, rank, options)
% Online non-negative matrix factorization (ONMF) algorithm.
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
%       S. S. Bucak, B. Gunsel,
%       "Incremental Subspace Learning via Non-negative Matrix Factorization,"
%       Pattern Recognition, 2009.
%    
%
% Created by H.Kasai and H.Sakai on Feb. 12, 2017

    m = size(V, 1);
    n = size(V, 2);  
 
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
    
    if ~isfield(options, 'rep_mode')
        rep_mode = 'fix';
    else
        rep_mode = options.rep_mode;
    end     
    
    if ~isfield(options, 'w_repeat')
        w_repeat = 1;
    else
        w_repeat = options.w_repeat;
    end 

    if ~isfield(options, 'h_repeat')
        h_repeat = 1;
    else
        h_repeat = options.h_repeat;
    end      
  
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end       
    
    if ~isfield(options, 'x_init')
        Wt = rand(m, rank);
        H = rand(rank, n);
    else
        Wt = options.x_init.W;
        H = options.x_init.H;
    end     

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end

 
    % initialize
    epoch = 0;
    R = zeros(m, n);
    %At = zeros(m, rank);
    %Bt = zeros(rank, rank);    
    grad_calc_count = 0;
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, Wt, H, R);
    infos.cost = f_val;
    optgap = f_val - f_opt;
    infos.optgap = optgap;   
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
    if verbose > 0
        fprintf('ONMF ACC: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end    
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end    
         
    % set start time
    start_time = tic();
    prev_time = start_time;
    
    % main outer loop
    %for epoch = 1 : max_epoch 
    while (optgap > tol_optgap) && (epoch < max_epoch)        
        
        % Reset sufficient statistic
        At = zeros(m, rank);
        Bt = zeros(rank, rank);        

        % main inner loop
        for t = 1 : batch_size : n - 1

            % Retrieve vt and ht
            vt = V(:, t:t + batch_size -1);
            ht = H(:, t:t+batch_size-1);

%             % uddate ht
%             Wtv = Wt.' * vt;
%             WtW = Wt.' * Wt;
%             for iii=1:h_repeat
%                 ht = ht .* (Wtv) ./ (WtW * ht);
%                 ht = ht + (ht<eps) .* eps;      
%             end
            
            Wtv = Wt.' * vt;
            WtW = Wt.' * Wt;
            if strcmp(rep_mode, 'adaptive')
                gamma = 1; 
                eps0 = 1; 
                j = 1;
                rhoh_alpha = rhoh*alpha;
                %while j <= floor(1+rhoh*alpha) &&  gamma >= delta*eps0
                while j <= rhoh_alpha && gamma >= delta*eps0
                    ht0 = ht;
                    ht = ht .* (Wtv) ./ (WtW * ht);
                    ht = ht + (ht<eps) .* eps;   
                    if j == 1
                        eps0 = norm(ht0-ht); 
                    end
                    gamma = norm(ht0-ht);  
                    j = j+1;
                end           
            else
                for iii=1:h_repeat
                    ht = ht .* (Wtv) ./ (WtW * ht);
                    ht = ht + (ht<eps) .* eps;      
                end                  
            end 
            
            
            % update sufficient statistics
            At = At + vt *  ht';
            Bt = Bt + ht *  ht';              

            % update W
            for iii=1:w_repeat
                Wt = Wt .* At ./ (Wt * Bt); 
                Wt = Wt + (Wt<eps) .* eps;
            end

            % store new h
            H(:,t:t+batch_size-1) = ht;  
            
            grad_calc_count = grad_calc_count + m * batch_size;
        end

        % calculate cost and optgap
        f_val = nmf_cost(V, Wt, H, R);
        optgap = f_val - f_opt; 
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
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
            if ~mod(epoch,disp_freq)
                fprintf('ONMF ACC: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end  
        
        prev_time = elapsed_time;
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end     
    
    x.W = Wt;
    x.H = H;
end





