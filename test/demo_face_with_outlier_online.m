function demo_face_with_outlier_online()
% Demonstation file for robust online NMF for face images with outlier
%
% Created by H.Kasai on Apr. 18, 2018

    clc;
    clear;
    close all;


    %% set parameters
    max_epoch   = 500;
    batch_size  = 100;
    N           = 1000;    
    K           = 49;    
    outlier_rho = 0.2; % set outlier level


    %% load data
    V = importdata('../data/CBCL_Face.mat');
    V = V(:,1:N);
    V = normalization(V, 50);
    F = size(V,1);
    [V, ~] = add_outlier(outlier_rho, F, N, V); % add outliers

    
    %% perform factroization
    options.max_epoch = max_epoch;
    options.batch_size = batch_size;
    options.verbose = 2;
    options.lambda = 1;
    
    [w_smu_nmf, infos_smu_nmf] = smu_nmf(V, K, options);
    
    options.accel = 1;
    options.repeat_inneriter = 5; 
    options.robust = true;
    options.max_epoch = floor(max_epoch / (options.repeat_inneriter + 1)); 
    [w_svrmu_nmf, infos_svrmu_nmf] = svrmu_nmf(V, K, options);        


    %% plot
    display_graph('grad_calc_count','cost', {'SMU', 'RSVRMU-ACC'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});
    display_graph('grad_calc_count','cost', {'SMU', 'RSVRMU-ACC'}, {w_smu_nmf, w_svrmu_nmf}, {infos_smu_nmf, infos_svrmu_nmf});
    
    
    %% display basis elements obtained with different algorithms
    plot_dictionnary(w_smu_nmf.W, [], [7 7]); 
    plot_dictionnary(w_svrmu_nmf.W, [], [7 7]); 
    
end