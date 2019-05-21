function [x, infos] = nmf_hals_so(V, rank, in_options)
% Orthogonal multiplicative updates (MU) for non-negative matrix factorization with soft orthogonal constraint (NMF-HALS-SO).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2,
%           where 
%           {V, W, H} >= 0, and W is orthogonal.
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
%           orth    : orthogonality 
%
% References
%           M. Shiga, K. Tatsumi, S. Muto, K. Tsuda, Y. Yamamoto, T. Mori, and T. Tanji, 
%           "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
%           Ultramicroscopy, Vol.170, p.43-59, 2016.
%   
%
% Originally created by Motoki Shiga, Gifu University, Japan.
% Modified by H.Kasai on Jul. 25, 2018
%
% Change log: 
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.norm_w    = 0;
    local_options.wo        = 0.1;  % weight of orthogonal constraints for W
    local_options.myeps     = 1e-16; % 2.2204e-16
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);  
    
    if options.verbose > 0
        fprintf('# HALS-SO: started ...\n');           
    end     
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;
    R = init_factors.R;
    
    % initialize
    epoch = 0;    
    grad_calc_count = 0; 
    
    wj = sum(W, 2);

    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
   
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, [], epoch, grad_calc_count, 0);
    orth_val = norm(W'*W - eye(rank),'fro');
    [infos.orth] = orth_val;

    if options.verbose > 1
        fprintf('HALS-SO: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)           
        
        % update W
        VHT = V*H';   
        HHT = H*H';
        for j = 1:rank
            wj     = wj - W(:,j);
            W(:,j) = VHT(:,j) - W*HHT(:,j) + HHT(j,j)*W(:,j);
            W(:,j) = W(:,j) - options.wo*(wj'*W(:,j))/(wj'*wj)*wj;
            %replace negative values with zeros (tiny positive velues)
            W(:,j) = max( (W(:,j) + abs(W(:,j)))/2, options.myeps);
            W(:,j) = W(:,j) / sqrt(W(:,j)'*W(:,j)); %normalize
            wj     = wj + W(:,j);
        end

        % update H
        VTW = V'*W;   
        WWT = W'*W;
        for j = 1:rank
            H(j,:) = VTW(:,j)' - WWT(:,j)'*H + WWT(j,j)*H(j,:);
            %replace negative values with zeros (tiny positive velues)
            H(j,:) = max( (H(j,:) + abs(H(j,:)))/2, options.myeps);
        end

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time);  
        orth_val = norm(W'*W - eye(rank),'fro');
        [infos.orth] = [infos.orth orth_val];
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('HALS-SO: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# HALS-SO: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# HALS-SO: Max epoch reached (%g).\n', options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;
end
