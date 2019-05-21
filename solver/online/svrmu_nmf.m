function [x, infos] = svrmu_nmf(V, rank, in_options)
% Stochastic variance reduced multiplicative update for non-negative matrix factorization (SVRMU-NMF) algorithm.
%
% Inputs:
%       matrix      V
%       rank        rank
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
%    
% Created by H.Kasai on Mar. 22, 2017
%
% Change log: 
%
%   Mar. 15, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.repeat_inneriter  = 1;
    local_options.W_sub_mode        = 'STD';    
    local_options.H_sub_mode        = 'STD';
    local_options.accel             = false;
    local_options.ls                = false;
    local_options.precon            = false;    
    local_options.h_repeat          = 1;
    local_options.rep_mode          = 'fix';
    local_options.stepsize_ratio    = 1; % stepsize ratio
    local_options.robust            = false;
    local_options.lambda            = 1;
    local_options.tol_optgap        = 1e-2;
    local_options.x_init_robust     = false;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    Wt = init_factors.W;
    H = init_factors.H;  
    R = init_factors.R;    

    % determine sub_mode
    if options.accel
        options.H_sub_mode = 'ACC';
    else
        if options.ls
            options.H_sub_mode = 'LS';
        else
            options.H_sub_mode = 'STD';  
        end
        options.h_repeat = 1;
    end
    
    if options.precon
        options.W_sub_mode = 'Precon';
        fprintf('Unfortunately, Precon variant does not work well.\n');
    else
        options.W_sub_mode = 'STD';
    end
    
    if options.robust
        mode = 'R-SVRMU-NMF';
    else
        mode = 'SVRMU-NMF';        
        R = zeros(m, n);
    end 
    
    if options.verbose > 0
        fprintf('# %s (%s,%s): started ...\n', mode, options.W_sub_mode, options.H_sub_mode);           
    end     
    
    if strcmp(options.rep_mode, 'adaptive')
        rhoh = 1+(m+m*rank)/(1*(rank+1));
        alpha = 2;
        delta = 0.01;       
    end    
   
    % initialize
    epoch = 0;  
    l = zeros(m, options.batch_size) + options.lambda;      
    grad_calc_count = 0;
        
    % permute samples
    if options.permute_on
        perm_idx = randperm(n);
    else
        perm_idx = 1:n;
    end   
    V = V(:,perm_idx);
    H = H(:,perm_idx);       

    % select disp_freq 
    disp_freq = set_disp_frequency(options);         
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1    
        fprintf('%s (%s,%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', mode, options.W_sub_mode, options.H_sub_mode, f_val, optgap); 
    end
         
    % initialize elapsed_time
    elapsed_time = 0;
    
    
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)   
        % set start time
        start_time = tic();
        
        % store W and H, and calculate full grad
        W_0 = Wt;
        H_0 = H;
        if ~options.robust
            W0_H0_H0T =  W_0  * (H_0 * H_0')/n;
        else
            R_0 = R;
            W0_H0_H0T =  (W_0  * H_0 + R_0 ) * H_0'/n;            
        end
        V_H0T = V * H_0'/n;
        
        if strcmp(options.W_sub_mode, 'Precon')        
            invH0H0t = inv(H_0 * H_0');
            invH0H0t = max(invH0H0t,0);        
        end
        
        grad_calc_count = grad_calc_count + m * n;
        

        % main inner loop
        for s = 1 : options.repeat_inneriter
            for t = 1 : options.batch_size : n - 1

                % Retrieve vt and ht
                start_idx = t;
                end_idx = t+options.batch_size-1;
                vt = V(:, start_idx:end_idx);
                ht = H(:, start_idx:end_idx);

                ht_0 = H_0(:, start_idx:end_idx);

                if ~options.robust

                    % uddate ht
                    Wtv = Wt' * vt;
                    WtW = Wt' * Wt;
                    if strcmp(options.H_sub_mode, 'ACC')
                        if strcmp(options.rep_mode, 'adaptive')
                            gamma = 1; 
                            eps0 = 1; 
                            j = 1;
                            rhoh_alpha = rhoh*alpha;
                            %while j <= floor(1+rhoh*alpha) &&  gamma >= delta*eps0
                            ht0 = ht;                        
                            while j <= rhoh_alpha && gamma >= delta*eps0
                                ht = ht .* (Wtv) ./ (WtW * ht);   
                                ht = ht + (ht<eps) * eps; 
                                if j == 1
                                    eps0 = norm(ht0-ht); 
                                end
                                gamma = norm(ht0-ht);  
                                j = j+1;
                            end           
                        else
                            for iii=1:options.h_repeat
                                ht = ht .* (Wtv) ./ (WtW * ht);                            
                                ht = ht + (ht<eps) * eps; 
                            end                  
                        end 
                    elseif strcmp(options.H_sub_mode, 'LS')
                        ht = calc_nls_nmf(vt, Wt, 1e-16);
                        ht = ht + (ht<eps) * eps;
                    else
                        ht = ht .* (Wtv) ./ (WtW * ht); 
                        ht = ht + (ht<eps) * eps; 
                    end        

                    % update W   
          
                    if strcmp(options.W_sub_mode, 'Precon')   
                        invhht = inv(ht * ht');
                        invhht = max(invhht,0);
                        invh0h0t = inv(ht_0 * ht_0');
                        invh0h0t = max(invh0h0t,0);
                        %Delta_minus = (W_0 * (ht_0 * ht_0') * invh0h0t + vt * ht' * invhht)/options.batch_size + V_H0T * invH0H0t;            
                        %Delta_plus  = (Wt * (ht * ht') * invhht + vt * ht_0' * invh0h0t)/options.batch_size + W0_H0_H0T * invH0H0t;  
                        Delta_minus = (W_0 * (ht_0 * ht_0') * invhht + vt * ht' * invhht)/options.batch_size + V_H0T * invhht;            
                        Delta_plus  = (Wt * (ht * ht') * invhht + vt * ht_0' * invhht)/options.batch_size + W0_H0_H0T * invhht;                         
                    else
                        Delta_minus = (W_0 * (ht_0 * ht_0') + vt * ht')/options.batch_size + V_H0T;            
                        Delta_plus  = (Wt * (ht * ht') + vt * ht_0')/options.batch_size + W0_H0_H0T;          
                    end
                else
                    
                    rt = R(:, start_idx:end_idx);                        
                    rt_0 = R_0(:, start_idx:end_idx);                     
                    
                    % uddate ht
                    Wtv = Wt' * vt;
                    if strcmp(options.H_sub_mode, 'ACC')
                        if strcmp(options.rep_mode, 'adaptive')
                            gamma = 1; 
                            eps0 = 1; 
                            j = 1;
                            rhoh_alpha = rhoh*alpha;
                            %while j <= floor(1+rhoh*alpha) &&  gamma >= delta*eps0
                            ht0 = ht;                        
                            while j <= rhoh_alpha && gamma >= delta*eps0
                                Wh_r = Wt * ht + rt;
                                ht = ht .* (Wtv) ./ (Wt' * Wh_r);   
                                ht = ht + (ht<eps) * eps; 
                                rt = rt .* vt ./ (Wh_r + l);  
                                if j == 1
                                    eps0 = norm(ht0-ht); 
                                end
                                gamma = norm(ht0-ht);  
                                j = j+1;
                            end           
                        else
                            for iii=1:options.h_repeat
                                Wh_r = Wt * ht + rt;
                                ht = ht .* (Wtv) ./ (Wt' * Wh_r);   
                                ht = ht + (ht<eps) * eps; 
                                rt = rt .* vt ./ (Wh_r + l);                              
                            end                  
                        end          
                    else
                        Wh_r = Wt * ht + rt;
                        ht = ht .* (Wtv) ./ (Wt' * Wh_r); 
                        ht = ht + (ht<eps) * eps; 
                        rt = rt .* vt ./ (Wh_r + l);                      
                    end        

                    % update W   
                    Delta_minus = ((W_0 * ht_0 + rt_0) * ht_0' + vt * ht')/options.batch_size + V_H0T;            
                    Delta_plus  = ((Wt * ht + rt) * ht' + vt * ht_0')/options.batch_size + W0_H0_H0T;                    
                end
                    
                if options.stepsize_ratio == 1
                    Wt = Wt .* (Delta_minus ./ Delta_plus);
                else
                    Wt = (1-options.stepsize_ratio) * Wt + options.stepsize_ratio * Wt .* (Delta_minus ./ Delta_plus);                    
                end
                
                Wt = Wt + (Wt<eps) * eps;


                % store new h
                H(:, start_idx:end_idx) = ht; 
                
                % Update R                
                if options.robust
                    R(:, start_idx:end_idx) = rt;            
                end

                grad_calc_count = grad_calc_count + m * options.batch_size;
            end
        end
        
        % measure elapsed time
        elapsed_time = elapsed_time + toc(start_time); 
        
        % update epoch
        epoch = epoch + 1;          

        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)            
                fprintf('%s (%s,%s): Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', mode, options.W_sub_mode, options.H_sub_mode, epoch, f_val, optgap, elapsed_time);
            end
        end  
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# %s (%s,%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', mode, options.W_sub_mode, options.H_sub_mode, f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# %s (%s,%s): Max epoch reached (%g).\n', mode, options.W_sub_mode, options.H_sub_mode, options.max_epoch);
        end    
    end
    
    x.W = Wt;
    x.H(:,perm_idx) = H;
    x.R = R;   
    
end





