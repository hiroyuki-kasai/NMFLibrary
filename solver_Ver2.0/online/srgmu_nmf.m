function [x, infos] = srgmu_nmf(V, rank, in_options)
% Stochastic recursive gradient multiplicative update for non-negative matrix factorization (SRGMU-NMF) algorithm.
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
% This file is part of NMFLibrary.
%
% Created by H.Kasai on March 22, 2017
%
% Change log: 
%
%       Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    m = size(V, 1);
    n = size(V, 2);

    % set local options
    local_options.repeat_inneriter    = 5;
    local_options.sub_mode            = 'STD';
    local_options.accel               = false;
    local_options.h_repeat            = 1;
    local_options.rep_mode            = 'fix';
    local_options.stepsize_ratio      = 1; % stepsize ratio
    local_options.robust              = false;
    local_options.lambda              = 1;

    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 


    if options.accel
        options.sub_mode = 'ACC';
    else
        options.sub_mode = 'STD';  
        options.h_repeat = 1;
    end
    
    if strcmp(options.rep_mode, 'adaptive')
        rhoh = 1+(m+m*rank)/(1*(rank+1));
        alpha = 2;
        delta = 0.01;       
    end 

    if ~isfield(options, 'x_init')
        Wt  = rand(m, rank);
        H   = rand(rank, n);
        R   = rand(m, n);          
    else
        Wt  = options.x_init.W;
        H   = options.x_init.H;
        R   = options.x_init.R;           
    end     

    if options.robust
        mode = 'R-SRGMU-NMF';
    else
        mode = 'SRGMU-NMF';        
        R = zeros(m, n);
    end     
   
    % initialize
    method_name = fprintf('%s (%s)', mode, options.sub_mode);    
    epoch = 0; 
    grad_calc_count = 0;    
    l = zeros(m, options.batch_size) + options.lambda;      
    
    if options.verbose > 0
        fprintf('# %s (%s): started ...\n', mode, options.sub_mode);           
    end    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1    
        fprintf('%s (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', mode, options.sub_mode, f_val, optgap); 
    end
         
    % set start time
    start_time = tic();
    
    % initialize
    %ht_prev = zeros(rank, options.batch_size);
    if options.robust
        rt_prev = zeros(m, options.batch_size);
    end
    
    % main outer loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end
        
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
        
        Wt_prev = Wt;
        
        Delta_plus = W0_H0_H0T;
        Delta_minus = V_H0T;
        
        grad_calc_count = grad_calc_count + m * n;
        

        % main inner loop
        for s = 1 : options.repeat_inneriter
            for t = 1 : options.batch_size : n - 1

                % Retrieve vt and ht
                start_idx = t;
                end_idx = t+options.batch_size-1;
                vt = V(:, start_idx:end_idx);
                ht = H(:, start_idx:end_idx);

                if ~options.robust

                    % uddate ht
                    Wtv = Wt' * vt;
                    WtW = Wt' * Wt;
                    Wtv_prev = Wt_prev' * vt;
                    WtW_prev = Wt_prev' * Wt_prev;  
                    ht_prev = ht;
                    if strcmp(options.sub_mode, 'ACC')
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
                                ht_prev = ht_prev .* (Wtv_prev) ./ (WtW_prev * ht_prev); 
                                ht_prev = ht_prev + (ht_prev<eps) * eps;                                  
                            end                  
                        end          
                    else
                        ht = ht .* (Wtv) ./ (WtW * ht); 
                        ht = ht + (ht<eps) * eps; 
                        ht_prev = ht_prev .* (Wtv_prev) ./ (WtW_prev * ht_prev); 
                        ht_prev = ht_prev + (ht_prev<eps) * eps;                         
                    end 
                    
                    %ht_prev = ht;

                    % update W   
                    Delta_minus = (Wt_prev * (ht_prev * ht_prev') + vt * ht')/options.batch_size + Delta_minus;            
                    Delta_plus  = (Wt * (ht * ht') + vt * ht_prev')/options.batch_size + Delta_plus;                    
                else
                    
                    rt = R(:, start_idx:end_idx);                        
                    rt_0 = R_0(:, start_idx:end_idx);                     
                    
                    % uddate ht
                    Wtv = Wt' * vt;
                    if strcmp(options.sub_mode, 'ACC')
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
                    
                
                Wt_prev = Wt;
                if options.stepsize_ratio == 1
                    Wt = Wt .* (Delta_minus ./ Delta_plus);
                else
                    Wt = (1-options.stepsize_ratio) * Wt + options.stepsize_ratio * Wt .* (Delta_minus ./ Delta_plus);                    
                end
                
                Wt = Wt + (Wt<eps) * eps;
                
                if options.robust
                    rt_prev = rt;
                end


                % store new h
                H(:, start_idx:end_idx) = ht; 
                
                % update R                
                if options.robust
                    R(:, start_idx:end_idx) = rt;            
                end

                grad_calc_count = grad_calc_count + m * options.batch_size;
            end
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);    
        
        % update epoch
        epoch = epoch + 1;          

        % store info
        infos = store_nmf_info(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display info
        display_info(method_name, epoch, infos, options);
        
    end
    
    x.W = Wt;
    x.H = H;
    x.R = R;   
    
end