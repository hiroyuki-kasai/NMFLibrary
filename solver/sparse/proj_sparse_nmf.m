function [x, infos] = proj_sparse_nmf(V, rank, in_options)
% Projection-based Sparse Nonnegative matrix factorization (Proj-Sparse-NMF)
%
% The problem of interest is defined as
%
%      min D(V||W*H) + lambda * sum(H(:)),
%      where 
%      {V, W, H} > 0, and 
%      the average Hoyer sparsity of the columns of W is given by s. 
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options  : options 
%           sW      : average sparsity of the columns of W. For sW=[] (default), 
%                       no sparsity constraint enforced adf. 
%           sH      : average sparsity of the rows of H. For sH=[] (default), 
%                       no sparsity constraint enforced. 
%           delta   : stopping criterion for inner iterations  -default=0.1
%           inneriter : maximum number of inner iterations when updating W
%                       and H. -default=10
%           FPGM    : Update of H using FPGM if options.FPGM = 1,
%                       (default:0 and HALS is used) 
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
%       Patrik O. Hoyer, 
%       "Non-negative matrix factorization with sparseness constraints," 
%       Journal of Machine Learning Research, vol.5, pp.1457-1469, 2004.
%
%       R. Ohib, N. Gillis, Niccol嘆 Dalmasso, S. Shah, V. K. Potluru, S. Plis,
%       "Explicit Group Sparse Projection with Applications to Deep Learning and NMF,"
%       arXiv preprint:1912.03896, 2019
%       
%
% This file is part of NMFLibrary.
%
%       This file has been ported from 
%       sparseNMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 15, 2022 (Hiroyuki Kasai): Ported initial version 
%
%       Jul. 14, 2022 (Hiroyuki Kasai): Fixed algorithm.
%

    
    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = [];
    local_options.FPGM = false;
    local_options.sW = 0;
    local_options.sH = 0;
    local_options.delta = 0.1;
    local_options.inneriter = 10;
    local_options.colproj = 0;  
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      

    % initialize
    method_name = 'Proj-Sparse';      
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end     

    % initialize for this algorithm
    % project sW/sH into [0,1] 
    options.sW = min(max(options.sW, 0), 1);
    options.sH = min(max(options.sH, 0), 1);    

    % projection W to achieve average sparsity s 
    if options.sW > 0
        W = weightedgroupedsparseproj_col(W, options.sW, options);
    end
    if options.sH > 0
        H = weightedgroupedsparseproj_col(H', options.sH, options);
        H = H'; 
    end    
    Wbest = W; 
    Hbest = H; 
    nV2 = sum(V(:).^2); 
    nV = sqrt(nV2); 
    e = [];   
     
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('Proj-Sparse-NMF: Epoch = 0000, cost = %.16e, optgap = %.4e\n', f_val, optgap); 
    end  
    
    % set start time
    start_time = tic();    

    % main loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end           

        % Normalize W and H so that columns/rows have the same norm, that is, 
        % ||W(:,k)|| = ||H(k,:)|| for all k 
        normW = sqrt((sum(W.^2))) + 1e-16; 
        normH = sqrt((sum(H'.^2))) + 1e-16; 
        for k = 1 : rank
            W(:,k) = W(:,k) / sqrt(normW(k)) * sqrt(normH(k)); 
            H(k,:) = H(k,:) / sqrt(normH(k)) * sqrt(normW(k)); 
        end
        
        % update H: 
        clear nnls_options;         

        if options.FPGM == 0 && options.sH == 0 
            % (1) No sparsity constraints: block coordinate descent method; 
            % See N. Gillis and F. Glineur, "Accelerated Multiplicative Updates 
            % and Hierarchical ALS Algorithms for Nonnegative Matrix Factorization", 
            % Neural Computation 24 (4), pp. 1085-1105, 2012.            
            nnls_options.init = H; 
            nnls_options.verbose = 0;
            nnls_options.max_epoch = 10;
            nnls_options.delta = 0.1;
            [H, ~, ~] = nnls_solver(V, W, nnls_options); % BCD on the rows of H      
        else
            % (2) With sparsity constraints: fast projected gradient method (FGM) 
            % similar to NeNMF from 
            % Guan, N., Tao, D., Luo, Z., & Yuan, B. NeNMF: An optimal 
            % gradient method for nonnegative matrix factorization, IEEE 
            % Transactions on Signal Processing, 60(6), 2882-2898, 2012.
            nnls_options.s = options.sH; 
            nnls_options.colproj = options.colproj;             
            H = fastgradsparseNNLS(V',W',H',nnls_options); 
            H = H'; 
        end
        
        % update W: same as W
        clear nnls_options;        
        if options.sW == 0 % A-HALS
            nnls_options.init = W';
            nnls_options.verbose = 0;            
            nnls_options.max_epoch = 10;
            nnls_options.delta = 0.1;            
            [W,HHt,VHt] = nnls_solver(V', H', nnls_options); % BCD on the rows of H 
            W = W'; 
            VHt = VHt'; 
        else                   % FPGM 
            nnls_options.s = options.sW;
            nnls_options.colproj = options.colproj;             
            [W,VHt,HHt] = fastgradsparseNNLS(V,H,W,nnls_options); 
        end

        % Keep best iterate in memory as FGM is not guaranteed to be monotone
        e = [e sqrt( max(0, (nV2-2*sum(sum(W.*VHt))+ sum(sum(HHt.*(W'*W)))) ) )/nV]; 
        if epoch >= 2 
            if e(end) <= e(end-1)
                Wbest = W; 
                Hbest = H; 
            end
        end        
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time);            
        
        % display info
        display_info('Proj-Sparse-NMF', epoch, infos, options);   

    end  

    x.W = Wbest;
    x.H = Hbest;
    
end