function [x, infos] = sep_symm_nmtf(V, rank, in_options)
% Separable symmetric nonnegative matrix tri-factorization (Sep-Symm-NMTF)
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
%       Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, Zhu, 
%       "A practical algorithm for topic modeling with provable guarantees,"
%       International Conference on Machine Learning (ICML), pp. 280-288, 
%       2013.
%
%
% This file is part of NMFLibrary.
%
% This file has been ported from 
%       septrisymNMF.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       June 14, 2022 (Hiroyuki Kasai): Ported initial version 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.delta = 1e-6;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);

    method_name = 'Sep-Symm-NMTF';
    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      

    
    % idendity K such that V(K,K) = W(K,:) S W(K,:)^T 
    spa_options.normalize = 1; 
    spa_sol = spa(V, rank, spa_options); 
    
    
    % solve V(K,K) z = q = V(K,:) 
    q = V(:, spa_sol.K)' * ones(m,1); 
    % y = A(K,K)\q; % This works in noiseless conditions
    options.alg_name = method_name;
    [y, WtW, WtV, infos] = nnls_solver(q, V(spa_sol.K, spa_sol.K), options); 
    fprintf('\n'); 

    % recover S and W 
    S = diag(y) * V(spa_sol.K, spa_sol.K) * diag(y); 
    % W = V(:,K)/( diag(z)*V(K,K) ); % This works in noiseless conditions
    [W, WtW, WtV, infos] = nnls_solver(V(:,spa_sol.K)', (diag(y) * V(spa_sol.K, spa_sol.K))', options); 
    fprintf('\n');     

    if options.verbose > 0
        fprintf('# %s: finished.\n', options.alg_name);           
    end       


    %% store results
    x.S = S;
    x.W = W';
   
end