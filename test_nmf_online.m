function test_nmf_online()
%
% demonstration file for SVRMU.
%
% This file illustrates how to use this library. 
% This demonstrates stochastic multiplicative updates (SMU) algorithm and 
% stochastic variance reduced multiplicative updates (SVRMU) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Apr. 05, 2018

    clc;
    clear;
    close all;

    %% generate synthetic data of (mxn) matrix       
    F = 300;
    N = 1000;
    V = rand(F,N);
    
    
    %% Initialize of rank to be factorized
    K = 5;
    
    
    %% Calculate f_opt
    fprintf('Calculating f_opt by HALS ...\n');
    options.alg = 'hals';
    options.max_epoch = 100;
    [w_sol, ~] = nmf_als(V, K, options);
    f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
    fprintf('Done.. f_opt: %.16e\n', f_opt);    


    %% perform factroization
    options.batch_size = N/10;
    options.verbose = 2;
    options.f_opt = f_opt;
    
    %
    [w_smu_nmf, infos_smu_nmf] = smu_nmf(V, K, options);
    [w_svrmu_nmf, infos_svrmu_nmf] = svrmu_nmf(V, K, options);    

    
    %% plot
    display_graph('epoch','optimality_gap', {'SMU', 'SVRMU'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});
    display_graph('time','optimality_gap', {'SMU', 'SVRMU'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});
    
    display_graph('grad_calc_count','optimality_gap', {'SMU', 'SVRMU'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});
    display_graph('grad_calc_count','optimality_gap', {'SMU', 'SVRMU'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});    
    
end


