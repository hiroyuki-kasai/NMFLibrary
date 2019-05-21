function [x, infos] = inmf(V, rank, in_options)
% Incremental non-negative matrix factorization (NMF) with outliers (INCNMF) algorithm.
%
% Inputs:
%       matrix      V
%       rank        rank
%       in_options  options
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
% Created by H.Kasai on Feb. 12, 2017
%
% Change log: 
%
%   Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);

    % set local options
    local_options = [];
    local_options.max_inneriter     = 1;
    local_options.online            = 0;
    local_options.tolcostdegrease   = 1e-8;
    local_options.alpha             = 0.5;  
    local_options.beta              = 0.5;     
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    if options.verbose > 0
        fprintf('# INMF: started ...\n');           
    end
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H; 
    R_zero = init_factors.R;     

    % initialize
    epoch = 0;
    grad_calc_count = 0;
    
    if options.online
        ht = H(:, end); %why
    else
        % Do nothing
    end
    A = V * H';
    B = H * H';
    R = zeros(m, n); 
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);       
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('INMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end
    
    % set start time
    start_time = tic();
    prev_time = start_time;
     
    % main outer loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)    
        
        % main inner loop
        for t = 1 : options.batch_size : n - 1
              
            vt = V(:, t:t+options.batch_size -1);
            
            % Need to be considered more carefully
            if options.online
                % Do nothing (??)
            else
                ht = H(:, t:t+options.batch_size-1);
            end
            
            for j = 1 : options.max_inneriter
                
                % Compute new ht (column vector at t)
                ht = ht .* (W' * vt) ./ (W' * (W * ht) + 1e-9); 
                ht = ht + (ht<eps) .* eps; 

                % Compute new W
                vht = vt * ht';
                hht = ht * ht';
                A_tmp = options.beta * A + options.alpha * vht;
                B_tmp = options.beta * B + options.alpha * hht;
                W = W .* ( A_tmp ./ (W * B_tmp + 1e-9) ); % Add 1e-9 to avoid 0 in the denom.
                W = W + (W<eps) .* eps;
                
               if j > 1
                    oldobj = newobj;
                end
                newobj = ((sum(sum((vt-W*ht).^2)))/m);
                
                if options.verbose > 2
                    fprintf('\t[%d-%d-%d] %e\n', epoch, t, j, newobj);
                end

                if j > 1 && (oldobj-newobj)/newobj < options.tolcostdegrease
                    break;
                end                

            end
            
            grad_calc_count = grad_calc_count + m * options.batch_size;            
            
            % update A and B
            A = options.beta * A + options.alpha * vht;
            B = options.beta * B + options.alpha * hht;     
            
            % update ht (columb of H at t)
            H(:,t:t+options.batch_size-1) = ht;             
            
            if options.verbose > 2
                % measure cost 
                f_val = nmf_cost(V, W, H, R);
                fprintf('INMF: inner [%03d-%03d] %e\n', epoch, t, f_val);
            end            
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % update epoch
        epoch = epoch + 1;    

        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)            
                fprintf('INMF: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', epoch, f_val, optgap, elapsed_time - prev_time);
            end
        end  
        
        prev_time = elapsed_time;
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# INMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, options.f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# INMF: Max epoch reached (%g).\n', options.max_epoch);
        end     
    end    
      
    x.W = W;
    x.H = H;
end


