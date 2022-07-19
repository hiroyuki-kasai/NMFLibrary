function ret = demo_python(rank)
%
% demonstration file for NMFLibrary.
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 19, 2022


    cd ..;
    run_me_first(false);
    cd demo_python/;

    % read a matrix file
    V = importdata('test.mat');
    delete test.mat    

    %% perform factroization
    options.verbose = 1;
    % MU
    options.alg = 'mu';
    [~, infos_mu] = fro_mu_nmf(V, rank, options);
    % Hierarchical ALS
    options.alg = 'hals';
    [~, infos_hals] = als_nmf(V, rank, options); 
    % Accelerated Hierarchical ALS
    options.alg = 'acc_hals';
    [~, infos_acchals] = als_nmf(V, rank, options);        
    
    ret = [];
    ret.mu = infos_mu;
    ret.hals = infos_hals;    
    ret.acchals = infos_acchals;      

end



