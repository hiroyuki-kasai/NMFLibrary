function demo_nenmf_clustering()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates a clustering application of NeNMF algorithms as well 
% as multiplicative updates (MU), Hierarchical ALS, and GNMF algorithms.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on May 21, 2019


    clc;
    clear;
    close all;

    rng('default');

    eval_num = 10;


    dataset = 'COIL20';
    %dataset = 'PIE';


    %% load CBCL face datasets
    fprintf('Loading %s dataset ... ', dataset);
    if strcmp(dataset, 'COIL20')
        load('../../data/COIL20.mat');	
        fea = AllSet.X';
        gnd = AllSet.y';    
        nClass = class_num;
    elseif strcmp(dataset, 'PIE')
        load('../../data/PIE_pose27.mat');	
        nClass = length(unique(gnd));    
    end
    fprintf('done\n');


    perm_idx = randperm(size(fea,1));
    fea = fea(perm_idx,:);
    gnd = gnd(perm_idx);

    % make dataset smaller for quick test
    % N = 500;
    % fea = fea(1:N,:);
    % gnd = gnd(1:N);


    % normalize each data vector to have L2-norm equal to 1 
    [fea, ~] = data_normalization(fea, [], 'std');

    [m, n] = size(fea');
    W0 = rand(m, nClass);
    H0 = rand(nClass, n);
    W0 = normalize_W(W0, 2);
    H0 = normalize_W(H0, 2);
    x_init.W = W0;
    x_init.H = H0; 


    %% Clustering in the original space
    rng('default');
    label = litekmeans(fea, nClass, 'Replicates', 20);
    mi = MutualInfo(gnd, label);
    purity = calc_purity(gnd, label);
    nmi = calc_nmi(gnd, label);
    fprintf('### k-means:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);            



    %% NMF
    options = [];
    options.kmeansInit = 0;
    options.maxIter = 100;
    options.nRepeat = 1;
    options.alpha = 0;
    %when alpha = 0, GNMF boils down to the ordinary NMF.
    rng('default');
    [~,V] = GNMF(fea', nClass, [], options, W0, H0'); %'
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### NMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 


    %% GNMF
    options = [];
    options.WeightMode = 'Binary';  
    sim_mat = constructW(fea, options);
    options.maxIter = 100;
    options.nRepeat = 1;
    options.alpha = 100;
    rng('default');
    [~,V] = GNMF(fea', nClass, sim_mat, options, W0, H0'); %'
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### GNMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 


    %% HALS
    options = [];
    options.init_alg = 'random';
    options.alg = 'hals';
    options.x_init = x_init;
    rng('default');
    [w_nmf_hals, infos_nmf_hals] = nmf_als(fea', nClass, options);   
    V = w_nmf_hals.H;
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### HALS:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 


    %% NeNMF
    options.lambda = 10;
    options.type = 'plain';
    rng('default');
    [w_nenmf_p, infos_nenmf_p] = nenmf(fea', nClass, options);
    V = w_nenmf_p.H;
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### NeNMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 

    options.lambda = 0.1;
    options.type = 'l1r';
    rng('default');
    [w_nenmf_l1, infos_nenmf_l1] = nenmf(fea', nClass, options);
    V = w_nenmf_l1.H;
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### NeNMF(L1R):\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 

    options.lambda = 100;
    options.type = 'l2r';
    rng('default');
    [w_nenmf_l2, infos_nenmf_l2] = nenmf(fea', nClass, options);
    V = w_nenmf_l2.H;
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### NeNMF(L2R):\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 

    options.lambda = 100;
    options.type = 'mr';
    options.sim_mat = sim_mat;
    rng('default');
    [w_nenmf_mr, infos_nenmf_mr] = nenmf(fea', nClass, options);
    V = w_nenmf_mr.H;
    rng('default');
    [accuracy] = eval_clustering_accuracy(V, gnd, nClass, eval_num);
    fprintf('### NeNMF(MR):\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', accuracy.nmi, accuracy.purity, accuracy.mi); 
    
end
    




