function [x, infos] = onmf_acc(V, rank, in_options)
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
%
% Change log: 
%
%   Feb. 12, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.rep_mode = 'fix';
    local_options.w_repeat = 1;
    local_options.h_repeat = 1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);   
    
    if options.verbose > 0
        fprintf('# ONMF ACC: started ...\n');           
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    Wt = init_factors.W;
    H = init_factors.H;  
    R = init_factors.R;
 
    % initialize
    epoch = 0;
    %At = zeros(m, rank);
    %Bt = zeros(rank, rank);    
    grad_calc_count = 0;
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('ONMF ACC: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end     
    
    % set start time
    start_time = tic();
    prev_time = start_time;
    
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)        
              
        % Reset sufficient statistic
        At = zeros(m, rank);
        Bt = zeros(rank, rank);        

        % main inner loop
        for t = 1 : options.batch_size : n - 1

            % Retrieve vt and ht
            vt = V(:, t:t+options.batch_size-1);
            ht = H(:, t:t+options.batch_size-1);            

%             % uddate ht
%             Wtv = Wt.' * vt;
%             WtW = Wt.' * Wt;
%             for iii=1:h_repeat
%                 ht = ht .* (Wtv) ./ (WtW * ht);
%                 ht = ht + (ht<eps) .* eps;      
%             end
            
            Wtv = Wt.' * vt;
            WtW = Wt.' * Wt;
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
                for iii=1:options.h_repeat
                    ht = ht .* (Wtv) ./ (WtW * ht);
                    ht = ht + (ht<eps) .* eps;      
                end                  
            end 
            
            
            % update sufficient statistics
            At = At + vt *  ht';
            Bt = Bt + ht *  ht';              

            % update W
            for iii=1:options.w_repeat
                Wt = Wt .* At ./ (Wt * Bt); 
                Wt = Wt + (Wt<eps) .* eps;
            end

            % store new h
            H(:,t:t+options.batch_size-1) = ht;  

            grad_calc_count = grad_calc_count + m * options.batch_size;            
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % update epoch
        epoch = epoch + 1;        
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, Wt, H, R, options, infos, epoch, grad_calc_count, elapsed_time);          

        % display infos
        if options.verbose > 0
            if ~mod(epoch,disp_freq)
                fprintf('ONMF ACC: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
        
        prev_time = elapsed_time;
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# ONMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# ONMF: Max epoch reached (%g).\n', options.max_epoch);
        end     
    end    
    
    x.W = Wt;
    x.H = H;
    x.R = R;
end





