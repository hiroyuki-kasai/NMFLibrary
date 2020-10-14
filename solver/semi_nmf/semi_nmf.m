function [x, infos] = semi_nmf(V, rank, in_options)
% Semi non-negative matrix factorization (Semi-NMF).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2,
%           where 
%           H > 0.
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options    
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x n)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
%
% Reference:
%       C.H.Q. Ding, T. Li, M. I. Jordan,
%       "Convex and semi-nonnegative matrix factorizations,"
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.1, 2010. 
%
%
% Modified by H.Kasai on Jul. 21, 2018
% Modified by H.Kasai on Apr. 22, 2019 (Bug fixed)
% Modified by H.Huangi on Oct. 14, 2020 (Bug fixed)

    
    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.max_iter  = 100;
    local_options.tolfun    = 1e-5;
    local_options.verbose   = 1; 
    %local_options.init_alg  = 'semi_random';
    local_options.init_alg  = 'LPinit';
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    if options.verbose > 0
        fprintf('# Semi-NMF: started ...\n');           
    end  
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    R_zero = zeros(m, n);
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);        
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('Semi-NMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  
    
    % set start time
    start_time = tic();    

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_iter)             

        % update W
        HHT = H * H';
        %W = V * H' * inv(HHT);  % Eq.(10)  % fixed by HH  
        W = V * H' / HHT;  % Eq.(10)  
        
        % update H
        VtW = V' * W;
        VtW_p = (abs(VtW)+VtW) ./ 2;  % Eq.(12)
        VtW_n = (abs(VtW)-VtW) ./ 2;  % Eq.(12)
        
        WtW = W' * W;
        WtW_p = (abs(WtW)+WtW) ./ 2;  % Eq.(12) % fixed by HH
        WtW_n = (abs(WtW)-WtW) ./ 2;  % Eq.(12) % fixed by HH         
        
        Ht = H' .* sqrt((VtW_p+H'*WtW_n) ./ max(VtW_n+H'*WtW_p,eps)); % Eq.(11)
        H = Ht';
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);            
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('Semi-NMF: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end         
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# Semi-NMF: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# Semi-NMF: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end    

    x.W = W;
    x.H = H;
end
