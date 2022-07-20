function ret = demo_python(V, rank, py_dict_options)
%
% demonstration file for NMFLibrary.
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 19, 2022

    % add path
    cd ..;
    run_me_first(false);
    cd demo_python/;

    % extract options
    in_options = struct(py_dict_options)



%     % set options
%     if isfield(in_options, 'verbose')
%         options.verbose = in_options.verbose;      
%     else
%         options.verbose = 1;  
%     end

    % set local options
    local_options = [];    
    local_options.verbose = 1;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);    

    %ret = 1;
    %return;


    %% perform factroization

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