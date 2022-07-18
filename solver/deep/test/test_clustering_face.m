function test_clustering_face()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates multiplicative updates (MU) algorithm and 
% hierarchical alternative least squares (Hierarchical ALS) algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Apr. 05, 2017

    %clc;
    clear;
    close all;

    if 0
        %% load CBCL face datasets
        dataset = importdata('../../../data/ORL_Face_img.mat');
        X_all = dataset.TrainSet.X;
        y_all = dataset.TrainSet.y;
        
        %% Initialize of rank to be factorized
        rank = 40;
        %rank_layers = [160 120 100 80 70 60 50 rank];
        rank_layers = [rank*2 rank];

        index = find(y_all <= rank);
        V = X_all(:,index);
        gnd = y_all(index);
        
    elseif 0
        dataset = importdata('../../../data/JAFFE.mat');
        X_all = dataset.AllSet.X;
        y_all = dataset.AllSet.y;
        
        %% Initialize of rank to be factorized
        rank = 10;
        %rank_layers = [160 120 100 80 70 60 50 rank];
        rank_layers = [rank*2 rank];

        index = find(y_all <= rank);
        V = X_all(:,index);
        gnd = y_all(index);        
        
        
    else
        % Yale Dataset -----------------------------
        dataset = importdata('../../../data/yale_mtv.mat');

        X{1,1} = NormalizeFea(dataset.X{1,1}, 0);
        X{1,2} = NormalizeFea(dataset.X{1,2}, 0);
        X{1,3} = NormalizeFea(dataset.X{1,3}, 0);
        V = X{1,1};
        gnd = double(dataset.gt);
        gnd = gnd';
        rank = 15;
        %rank_layers = [rank*5 rank];
        rank_layers = [rank*2 rank];
    end
    
    
    % Same preprocessing as Lee and Seung
%     Vorg = V;
%     V = V - mean(V(:));
%     V = V / sqrt(mean(V(:).^2));
%     V = V + 0.25;
%     V = V * 0.25;
%     V = min(V,1);
%     V = max(V,0);     
    

    % set options
    max_epoch = 100;
    eval_clustering_num = 20;
    
    

    %% NMF
    options = [];
    options.kmeansInit = 0;
    options.maxIter = max_epoch;
    options.nRepeat = 1;
    options.alpha = 0; % GNMF wotj alpha = 0 boils down to the ordinary NMF.

    [~,nmf_H] = GNMF(V, rank, [], options); %'
    %options.alg = 'mu';
    options.verbose = 2;
    options.max_epoch = max_epoch;        
    %[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options); 
    %nmf_H = w_nmf_mu.H;
    % Hierarchical ALS
    %options.alg = 'hals';
    %[w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);           
    %nmf_H = w_nmf_hals.H;

%         % Clustering in the NMF subspace
%         rand('twister', 5489);
%         label = litekmeans(nmf_H', rank, 'Replicates', 20);
%         mi = MutualInfo(gnd, label);
%         purity = calc_purity(gnd, label);
%         nmi = calc_nmi(gnd, label);
%         fprintf('### NMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);      

    [accuracy] = eval_clustering_accuracy(nmf_H, gnd', rank, eval_clustering_num);
    fprintf('### NMF:\tNMI:%5.4f, Purity:%5.4f, ACC:%5.4f, MutualInfo:%5.4f\n\n', accuracy.nmi, accuracy.purity, accuracy.acc, accuracy.mi);


    
    verbose = 1;

    %% Deep-Semi-NMF

    options.verbose = verbose;
    options.max_epoch = max_epoch;
    options.apg_maxiter = 10;
    options.updateH_alg = 'apg';
    options.gnd = gnd';
    options.classnum = rank;
    options.eval_clustering_num = 10;
    options.initialize_max_epoch = 100;
    [w_deep_semi_nmf, infos_deep_semi_nmf] = deep_semi_nmf(V, rank_layers, options);
    Hm = w_deep_semi_nmf.H{end};
    [accuracy] = eval_clustering_accuracy(Hm, gnd', options.classnum, eval_clustering_num);
    fprintf('### Deep-Semi-NMF:\tNMI:%5.4f, Purity:%5.4f, ACC:%5.4f, MutualInfo:%5.4f\n\n', accuracy.nmi, accuracy.purity, accuracy.acc, accuracy.mi);

    
    %% Deep-ns-NMF
    options.theta = 0.3;
    options.apg_maxiter = 10;
    %options.update_alg = 'mu';
    options.update_alg = 'apg';
    %options.max_epoch = max_epoch;
    options.gnd = gnd';
    options.classnum = rank;        
    options.verbose = verbose;
    %options.norm_w = 1;
    options.eval_clustering_num = 10;
    options.initialize_max_epoch = 100;
    [w_deep_ns_nmf, infos_deep_ns_nmf] = deep_ns_nmf(V, rank_layers, options); 
    Hm = w_deep_ns_nmf.H{end};
    
    [accuracy] = eval_clustering_accuracy(Hm, gnd', options.classnum, eval_clustering_num);
    fprintf('### Deep-ns-NMF:\tNMI:%5.4f, Purity:%5.4f, ACC:%5.4f, MutualInfo:%5.4f\n\n', accuracy.nmi, accuracy.purity, accuracy.acc, accuracy.mi);
     
     
     %% Bidirectional
    options.verbose = verbose;
    options.max_epoch = max_epoch;
    options.apg_maxiter = 10;
    options.updateH_alg = 'apg';
    options.gnd = gnd';
    options.classnum = rank;
    [w_deep_ns_bi_nmf, infos_deep_bi_ns_nmf] = deep_bidirectional_nmf(V, rank_layers, options);
    Hm = w_deep_ns_bi_nmf.H{end};
    %[w_deep_semi_nmf, infos_deep_semi_nmf] = semi_nmf(V, rank_layers, options);
    %Hm = w_deep_semi_nmf.H;
    rand('twister', 5489);
    label = litekmeans(Hm', rank, 'Replicates', 20);
    mi = MutualInfo(gnd, label');
    purity = calc_purity(gnd, label);
    nmi = calc_nmi(gnd, label');
    %nmi2 = nmi_onmf(gnd, label');
    fprintf('### Deep-Semi-NMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);

    %[acc, nmii, ~ ]= evalResults_multiview(Hm, gnd'); 


        
end


