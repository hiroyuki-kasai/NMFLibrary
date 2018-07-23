function [x, infos] = sparse_nmf(V, rank, in_options)
% Sparse Nonnegative matrix factorization (sparseNMF)
%
% The problem of interest is defined as
%
%           min KL(V - WH) + lambda * sum(H(:)),
%           where 
%           {V, W, H} > 0.
%
%       L1-based sparsity constraint on H.
%       Normalizes W column-wise.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
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
% Reference:
%
%
% Created by Patrik Hoyer, 2006 (and modified by Silja Polvi-Huttunen, University of Helsinki, Finland, 2014)
% Modified by H.Kasai on Jul. 23, 2018


    % set dimensions and samples
    m = size(V, 1);
    n = size(V, 2);

    % set local options 
    local_options.lambda = 0;   % regularizer for sparsity
    local_options.cost  = 'EUC'; % 'EUC' or 'KL'
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0; 
    
    if ~isfield(options, 'x_init')
        [W, H] = NNDSVD(abs(V), rank, 0);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end    
      

    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
   
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    % store additionally different cost
    infos.cost_new = sum(sum((V.*log(V./(W*H))) - V + W*H)) + options.lambda*sum(sum(H));
    
    if options.verbose > 1
        fprintf('sparseNMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) 

        % update H with MU
        %H = (H.*(W'*(V./(W*H))))/(1+alpha);
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        H = (H.*(W'*VC))/(1+options.lambda);
        
        sum(sum(H))
        
        % update W by Lee and Seung's divergence step
        %W = W.*((V./(W*H))*H')./(ones(vdim,1)*sum(H'));
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        W = W.*(VC*H')./(ones(m,1)*sum(H,2)');     
        
        % Liu, Zheng, and Lu add this normalization step
        W = W./(ones(m,1)*sum(W));        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time);  
        % store additionally different cost
        f_val_new = sum(sum((V.*log(V./(W*H))) - V + W*H)) + options.lambda*sum(sum(H));
        infos.cost_new = [infos.cost_new f_val_new];
        
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('sparseNMF: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end      

    end

    x.W = W;
    x.H = H;
end