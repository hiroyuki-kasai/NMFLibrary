function [x, infos] = sagmu_nmf(V, rank, in_options)
% Stochastic averaging gradient multiplicative update for non-negative matrix factorization (SAGMU-NMF) algorithm.
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
% Reference:
%       H. Kasai, 
%       "Accelerated stochastic multiplicative update with gradient averaging for nonnegative matrix factorizations," 
%       EUSIPCO, 2018.
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on March 22, 2017
%
%       Feb. 26, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    m = size(V, 1);
    n = size(V, 2);  
    
    
    % set local options
    local_options.fast_calc             = true;
    local_options.permute_on            = true;
    local_options.sub_mode              = 'STD';
    local_options.accel                 = false;
    local_options.ls                    = false;
    local_options.h_repeat              = 1;
    local_options.rep_mode              = 'fix';
    local_options.stepsize_ratio        = 1; % stepsize ratio
    local_options.robust                = false;
    local_options.lambda                = 1;

    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end      
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);     
 
    number_of_batches = floor(n/options.batch_size);
    
    if ~isfield(options, 'accel')
        options.accel = false;
        if options.ls
            options.sub_mode = 'LS';
        else
            options.sub_mode = 'STD';  
        end     
    else
        if options.accel
            options.sub_mode = 'ACC';
        else
            if options.ls
                options.sub_mode = 'LS';
            else
                options.sub_mode = 'STD';  
            end
        end
    end  
    
    if options.accel
        if ~isfield(options, 'h_repeat')
            options.h_repeat = 1;
        else
        end  
        
        if ~isfield(options, 'rep_mode')
            options.rep_mode = 'fix';
        else
        end
    else
        options.h_repeat = 1;
    end
    
    if options.h_repeat == 1
        if options.ls
            options.sub_mode = 'LS';
        else
            options.sub_mode = 'STD';  
        end
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
    
    % permute samples
    if options.permute_on
        perm_idx = randperm(n);
    else
        perm_idx = 1:n;
    end   
    V = V(:, perm_idx);
    H = H(:, perm_idx); 
    R = R(:, perm_idx);      

    if options.robust
        mode = 'R-SAGMU-NMF';
    else
        mode = 'SAGMU-NMF';        
        R = zeros(m, n);
    end
    
    % initialize
    method_name = sprintf('%s (%s)', mode, options.sub_mode);    
    epoch = 0;    
    l = zeros(m, options.batch_size) + options.lambda;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end    
    
    % prepare arrays for vht and Whht 
    vht = cell(number_of_batches,1);
    Whht = cell(number_of_batches,1);
    
    % store vht and Whht 
    cnt = 0;
    for t=1: options.batch_size : n - 1
        cnt = cnt + 1;
        vt = V(:,t:t+options.batch_size-1);
        ht = H(:,t:t+options.batch_size-1);
        rt = R(:,t:t+options.batch_size-1);        
        vht{cnt} = vt * ht';
        Whht{cnt} = (Wt * ht + rt) * ht';
    end   
    
    % prepare Delta_minus and Delta_plus
    if options.fast_calc
        Delta_minus = zeros(m, rank);
        Delta_plus = zeros(m, rank);     
    end 
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, options.sub_mode, f_val, optgap); 
    end     
   
         
    % set start time
    start_time = tic();
    
    % main outer loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end        

        
        cnt = 0;
        % main inner loop
        for t = 1 : options.batch_size : n - 1
            cnt = cnt + 1;

            % retrieve vt and ht
            vt = V(:,t:t+options.batch_size-1);
            ht = H(:,t:t+options.batch_size-1);
            
            
            if ~options.robust
            
                % uddate ht
                Wtv = Wt' * vt;
                WtW = Wt' * Wt;
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
                            ht = ht + (ht<eps) .* eps;   
                            if j == 1
                                eps0 = norm(ht0-ht); 
                            end
                            gamma = norm(ht0-ht);  
                            j = j+1;
                        end           
                    else
                        for iii=1:options.h_repeat
                            ht = ht .* (Wtv) ./ (WtW * ht);
                            ht = ht + (ht<eps) .* eps;   
                        end     
                        %ht = ht + (ht<eps) .* eps;
                    end
                elseif strcmp(options.sub_mode, 'LS')
                    ht = calc_nls_nmf(vt, Wt, 1e-16);
                    ht = ht + (ht<eps) * eps;                    
                else
                    ht = ht .* (Wtv) ./ (WtW * ht);
                    ht = ht + (ht<eps) .* eps;          
                end
            else
                
                rt = R(:, t:t+options.batch_size-1);
                
                % uddate ht
                Wtv = Wt' * vt;
                %WtW = Wt' * Wt;
                if strcmp(options.sub_mode, 'ACC')
                    if strcmp(options.rep_mode, 'adaptive')
                        gamma = 1; 
                        eps0 = 1; 
                        j = 1;
                        rhoh_alpha = rhoh*alpha;
                        %while j <= floor(1+rhoh*alpha) &&  gamma >= delta*eps0
                        ht0 = ht;                    
                        while j <= rhoh_alpha && gamma >= delta*eps0
    %                         ht = ht .* (Wtv) ./ (Wt' * (Wt * ht + rt));
    %                         ht = ht + (ht<eps) .* eps;   
    %                         rt = rt .* vt ./ (Wt * ht + rt + l);
                            Wh_r = Wt * ht + rt;
                            ht = ht .* (Wtv) ./ (Wt' * Wh_r);
                            ht = ht + (ht<eps) .* eps;   
                            rt = rt .* vt ./ (Wh_r + l); 
                            if j == 1
                                eps0 = norm(ht0-ht); 
                            end
                            gamma = norm(ht0-ht);  
                            j = j+1;
                        end           
                    else
                        for iii=1:options.h_repeat
    %                         ht = ht .* (Wtv) ./ (Wt' * (Wt * ht + rt));
    %                         ht = ht + (ht<eps) .* eps;
    %                         rt = rt .* vt ./ (Wt * ht + rt + l); 
                            Wh_r = Wt * ht + rt;
                            ht = ht .* (Wtv) ./ (Wt' * Wh_r);
                            ht = ht + (ht<eps) .* eps;   
                            rt = rt .* vt ./ (Wh_r + l);                        
                        end                  
                    end          
                else
    %                 ht = ht .* Wtv ./ (Wt' * (Wt * ht + rt));
    %                 ht = ht + (ht<eps) .* eps;
    %                 rt = rt .* vt ./ (Wt * ht + rt + l);
                    Wh_r = Wt * ht + rt;
                    ht = ht .* (Wtv) ./ (Wt' * Wh_r);
                    ht = ht + (ht<eps) .* eps;   
                    rt = rt .* vt ./ (Wh_r + l);                  
                end                
                
            end

            % store vht and Whht 
            Whht_org = Whht{cnt}; 
            vht_org = vht{cnt};
            
            % update
            vht{cnt} = vt * ht';
            if ~options.robust
                Whht{cnt} = Wt * (ht * ht');             
            else
                Whht{cnt} = (Wt * ht + rt) * ht'; 
            end

            if epoch > 0
                if options.fast_calc
                    delta_minus = Delta_minus/n + (vht{cnt} + Whht_org)/options.batch_size;
                    delta_plus = Delta_plus/n + (Whht{cnt} + vht_org)/options.batch_size;
                    
                    % update Delta_minus and Delta_plus
                    Delta_minus = Delta_minus + (vht{cnt} - vht_org);
                    Delta_plus = Delta_plus + (Whht{cnt} - Whht_org);                            
                else
                    delta_minus = zeros(m, rank);
                    delta_plus = zeros(m, rank);

                    for jj=1:number_of_batches
                        delta_minus = delta_minus + vht{jj};
                        delta_plus = delta_plus + Whht{jj};
                    end  
                    
                    delta_minus = delta_minus/n;
                    delta_plus = delta_plus/n;

                    delta_minus = delta_minus + (vht{cnt} + Whht_org)/options.batch_size;
                    delta_plus = delta_plus + (Whht{cnt} + vht_org)/options.batch_size;              
                end
                    
            else
                delta_minus = vht{cnt}/options.batch_size;
                delta_plus = Whht{cnt}/options.batch_size;  
                
                if options.fast_calc
                    % update Delta_minus and Delta_plus
                    Delta_minus = Delta_minus + vht{cnt};
                    Delta_plus = Delta_plus + Whht{cnt};                    
                end
            end
            
            % update W                
            if options.stepsize_ratio == 1
                Wt = Wt .* (delta_minus ./ delta_plus);
            else
                Wt = (1-ratio)* Wt + ratio * Wt .* (delta_minus ./ delta_plus);                    
            end

            Wt = Wt + (Wt<eps) .* eps;
            
            % store new h
            H(:,t:t+options.batch_size-1) = ht;  
            
            % update R
            if options.robust
                R(:,t:t+options.batch_size-1) = rt;            
            end
            
            grad_calc_count = grad_calc_count + m * options.batch_size;
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
    x.H(:,perm_idx) = H;
    x.R = R;

end