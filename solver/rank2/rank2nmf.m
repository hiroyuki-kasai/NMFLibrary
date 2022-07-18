function [x, infos] = rank2nmf(V, in_options)
% Rank-two NMF computes a rank-two NMF of X
%
%       If X >= 0 and rank(X) = 1, take W as any non-zero column of X and compute
%       H accordingly 
% 
%       If X >= 0 and rank(X) = 2, this is Algorithm 4.1 in Chapter 4, 
%       with alpha1 = alpha2 = 0. This is an exact algorithm. 
% 
%       Otherwise, the input matrix has is replaced by max(0,X2) where X2 is its 
%       best rank-two approximation; cf. Chapter 6. 
%
% Inputs:
%       matrix      V
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Nicolas Gillis,
%       "Nonnegative Matrix Factorization,"
%       SIAM, 2020.
%
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%       Rank2NMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 13, 2022 (Mitsuhiko Horie and Hiroyuki Kasai): Ported initial version 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.delta = 1e-6;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);
    
    if options.verbose > 0
        fprintf('# Rank2-NMF: started ...\n');           
    end 

    if min(V(:)) < 0 
        if options.verbose > 0      
            fprintf('  * Rank2-NMF: The input matrix is not nonnegative: V <-- max(V,0).\n'); 
        end
        V = max(0,V); 
    end 
    [u, s, v] = svds(V, 3);

    % Case 0. rank(V) = 0
    if s(1, 1) <= eps
        if options.verbose > 0          
            fprintf('  * Rank2-NMF: The input matrix is (close to) zero.');
        end
        [m,n] = size(V); 
        W = zeros(m,1); 
        H = zeros(1,n); 
        return;
    end

    % Case 1. rank(V) = 1
    if s(2, 2) < 1e-9 * s(1, 1)
        if options.verbose > 0           
            fprintf('  * Rank2-NMF: The input matrix has rank one, an optimal solution is returned.\n');
        end
        sumV = sum(V); 
        [~,j] = max(sumV); 
        W = V(:,j); 
        H = sumV/sum(W); 
        return; 
    end

    % Case 2. rank(V) >= 2 
    if s(3, 3) > 1e-9 * s(2, 2)
        if options.verbose > 0           
	        fprintf('  * Rank2-NMF: The input matrix does not have rank two.\n');  
            fprintf('  * Rank2-NMF: V <-- rank(V2,0) where V2 is the rank-2 truncated SVD of V.\n'); 
            fprintf('  * Rank2-NMF: The solution computed might not be optimal.\n'); 
        end
        [u, s, v] = svds(V, 2);
        Vj = max(u*s*v', 0); 
    else
        if options.verbose > 0   
            fprintf('  * Rank2-NMF: The input matrix has rank two, an optimal solution is returned.\n');
        end
        Vj = V; 
    end

    % remove zero columns
    sumVj = sum(Vj); 
    J = find(sumVj > 0); 
    Vj = Vj(:, J);
    
    % normalize columns
    [m, n] = size(Vj); 
    for i = 1 : n
        Vj(:, i) = Vj(:, i) / sum(Vj(:, i));
    end
    
    % compute extreme points of conv(Vj) hence a feasible solution W
    % given that rank(V) = 2. 
    W = zeros(m, 2); 
    [~, j1] = max(sum(Vj.^2)); 
    W(:, 1) = Vj(:, j1); 
    [~, j2] = max(sum( (Vj-repmat(Vj(:, j1), 1, n)).^2)); 
    W(:, 2) = Vj(:, j2); 
    
    W = full(W); 
    WtW = W' * W;
    WtV = W' * V;
    
    % If no initial matrices are provided, H is initialized as follows: 
    if ~isfield(options,'init') || isempty(options.init)
        H = nnls_init_nmflibrary(V, W, WtW, WtV); 
    else
        H = options.init; 
    end 

    
    %% perform main routine
    options.alg_name = 'Rank2-NMF';
    options.V = V;
    options.W = W;
    options.main_routine_mode = true;
    options.eps0 = 0; 
    options.eps = 1;

    [H, WtW, WtV, infos] = nnls_solver(V, W, options); 

    x.W = W;
    x.H = H;
   
end