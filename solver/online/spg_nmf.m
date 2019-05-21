function [x, infos] = spg_nmf(V, rank, in_options)
% Stochastic projected gradient for non-negative matrix factorization (SMU-NMF) algorithm.
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
% Created by H.Kasai on Mar. 28, 2017
%
% Change log: 
%
%   Mar. 14, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.W_sub_mode          = 'STD';
    local_options.H_sub_mode          = 'LS';
    local_options.accel               = false;
    local_options.ls                  = false;  
    local_options.precon              = false;      
    local_options.h_repeat            = 1;
    local_options.rep_mode            = 'fix';
    local_options.robust              = false;

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
    
    
    if options.verbose > 0
        fprintf('# %s (%s): started ...\n', 'SPG-NMF', options.H_sub_mode);           
    end     
 
    % permute samples
    if options.permute_on
        perm_idx = randperm(n);
    else
        perm_idx = 1:n;
    end   
    V = V(:,perm_idx);
    H = H(:,perm_idx);     

    % initialize
    epoch = 0;    
    grad_calc_count = 0;
    
    if strcmp(options.rep_mode, 'adaptive')
        rhoh = 1+(m+m*rank)/(1*(rank+1));
        alpha = 2;
        delta = 0.01;       
    end     
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);         
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('SPG-NMF (%s,%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.W_sub_mode, options.H_sub_mode, f_val, optgap); 
    end     
   
         
    % initialize elapsed_time
    elapsed_time = 0;
    
    nesterov_alpha = 1;

    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)   
        % set start time
        start_time = tic();        
        
        cnt = 0;
        % main inner loop
        for t = 1 : options.batch_size : n - 1
            cnt = cnt + 1;

            % retrieve vt and ht
            vt = V(:,t:t+options.batch_size-1);
            ht = H(:,t:t+options.batch_size-1);
            
            % uddate ht
            Wtv = Wt.' * vt;
            WtW = Wt.' * Wt;
            if strcmp(options.H_sub_mode, 'ACC')
                if strcmp(options.rep_mode, 'adaptive')
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
                    for ii=1:options.h_repeat            
                        ht = ht .* (Wtv) ./ (WtW * ht);
                        ht = ht + (ht<eps) .* eps;      
                    end                      
                end
            elseif strcmp(options.H_sub_mode, 'LS')
                ht = calc_nls_nmf(vt, Wt, 1e-16);
                ht = ht + (ht<eps) * eps;                
            else
                ht = ht .* (Wtv) ./ (WtW * ht);
                ht = ht + (ht<eps) .* eps;                  
            end

            % update W
            if strcmp(options.W_sub_mode, 'Precon')            
                Wt = projection_precon_mnls(vt, ht', Wt);
            elseif strcmp(options.W_sub_mode, 'Nesterov') 
                options.nesterov_maxit = 1;
                options.nesterov_func_type = 'smooth';
                %options.nesterov_func_type = 'stochastic';
                %options.nesterov_func_type = 'strong_alpha_beta';
                
                [Wt, apg_iter, nesterov_alpha] = nesterov_mnls(vt, ht', Wt, nesterov_alpha, options.nesterov_maxit, options.nesterov_func_type);
                %nesterov_alpha
            else
                Wt = projection_mnls(vt, ht', Wt);                
            end
            Wt = Wt + (Wt<eps) .* eps;

            
            % store new h
            H(:,t:t+options.batch_size-1) = ht;  
            
            grad_calc_count = grad_calc_count + m * options.batch_size;
        end

        % measure elapsed time
        elapsed_time = elapsed_time + toc(start_time); 
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch,disp_freq)
                fprintf('SPG-NMF (%s,%s): Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', options.W_sub_mode, options.H_sub_mode, epoch, f_val, optgap, elapsed_time);
            end
        end       
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# SPG-NMF (%s,%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.W_sub_mode, options.H_sub_mode, f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# SPG-NMF (%s,%s): Max epoch reached (%g).\n', options.W_sub_mode, options.H_sub_mode, options.max_epoch);
        end    
    end
    
    x.W = Wt;
    x.H(:,perm_idx) = H;
end







