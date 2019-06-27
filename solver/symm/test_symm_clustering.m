function test_symm_clustering()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates Symm-ANLS algorithm and Symm-Newton algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Jun. 26, 2019

    clc;
    clear;
    close all;

    %% generate synthetic data of (mxn) matrix  
    if 1
        input = importdata('../../data/ORL.mat'); 
        M = input.data;
        gnd = input.label;
    else
        input = importdata('../../data/COIL20.mat'); 
        M = (input.TrainSet.X)';
        gnd = (input.TrainSet.y)';
    end
    clear input;


    V = calcu_similarity_matrix(M);
    rank = length(unique(gnd)); 
    
    
    %W_init = 2 * full(sqrt(mean(mean(V)) / rank)) * rand(m, rank);
    %W_init = rand(m, rank);
    %options.x_init.W = W_init;
    %options.x_init.H = (options.x_init.W)';    
    

    lambda = 0.66;  

   
    
    %% Initialize of rank to be factorized
 
    options.verbose = 2;
    options.max_epoch = 100;
    options.calc_symmetry = true;
    options.calc_clustering_acc = true;
    options.clustering_gnd = gnd;
    options.clustering_classnum = rank;
    %options.clustering_eval_num = 10;
    %options.init_alg = 'symm_mean';


    %% perform factroization
    % Symm-ANLS
    options.alpha = lambda;
    [w_symm_anls, infos_symm_anls] = symm_anls(V, rank, options);
    % Symm-Newton
    [w_symm_newton, infos_symm_newton] = symm_newton(V, rank, options);
    % Symm-Hals
    options.lambda = lambda;
    [w_symm_halsacc, infos_symm_halsacc] = symm_halsacc(V, rank, options);      
    
    
    %% plot
    display_graph('epoch','cost', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});
    display_graph('time','cost', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});
    
    %% symmetry
    display_graph('epoch','symmetry', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});    
    
    %% clustering
    display_graph('epoch','clustering_acc', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});  
    display_graph('epoch','clustering_nmi', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc}); 
    display_graph('epoch','clustering_purity', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});     
    
end


