function test_display_face()
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

    clc;
    clear;
    close all;

    %% load CBCL face datasets
    %V = importdata('../../../data/CBCL_face_new.mat');
    V = importdata('../../../data/CBCL_face.mat');
    
    % Same preprocessing as Lee and Seung
    Vorg = V;
    V = V - mean(V(:));
    V = V / sqrt(mean(V(:).^2));
    V = V + 0.25;
    V = V * 0.25;
    V = min(V,1);
    V = max(V,0); 
    
    list=randperm(size(V,2));
    V=V(:,list);    

    
    %% Initialize of rank to be factorized
    rank_layers = [49 25 16];
    
    % set options
    options.max_epoch = 100;
    options.verbose = 2;


    %% Deep-Semi-NMF
    [w_deep_semi_nmf, infos_deep_semi_nmf] = deep_semi_nmf(V, rank_layers, options);
    semiZ1 = w_deep_semi_nmf.Z{1};
    semiZ2 = semiZ1 * w_deep_semi_nmf.Z{2};
    semiZ3 = semiZ2 * w_deep_semi_nmf.Z{3};
    semiW1 = semiZ1./(ones(size(semiZ1,1),1)*sqrt(sum(semiZ1.^2))); 
    semiW2 = semiZ2./(ones(size(semiZ2,1),1)*sqrt(sum(semiZ2.^2))); 
    semiW3 = semiZ3./(ones(size(semiZ3,1),1)*sqrt(sum(semiZ3.^2)));     

 
    
    %% Deep-ns-NMF
    options.theta = 0.5;
    %options.update_alg = 'mu';
    options.update_alg = 'apg';
    options.apg_maxiter = 10;
    [w_deep_ns_nmf, infos_deep_ns_nmf] = deep_ns_nmf(V, rank_layers, options);    
    nsZ1 = w_deep_ns_nmf.Z{1};
    nsZ2 = nsZ1 * w_deep_ns_nmf.Z{2};
    nsZ3 = nsZ2 * w_deep_ns_nmf.Z{3};  
    nsW1 = nsZ1./(ones(size(nsZ1,1),1)*sqrt(sum(nsZ1.^2))); 
    nsW2 = nsZ2./(ones(size(nsZ2,1),1)*sqrt(sum(nsZ2.^2))); 
    nsW3 = nsZ3./(ones(size(nsZ3,1),1)*sqrt(sum(nsZ3.^2)));       

    
    %% Plotting
    display_graph('iter','cost', {'Deep-SemiNMF', 'Deep-nsNMF'}, {w_deep_semi_nmf, w_deep_ns_nmf}, {infos_deep_semi_nmf, infos_deep_ns_nmf});
    
    % display basis
    


%     plot_dictionnary(V(:,index_ext), [], [7 7]);
%     plot_dictionnary(W1, [], [7 7]);
%     plot_dictionnary(W2, [], [5 5]);
%     plot_dictionnary(W3, [], [4 4]); 
    
    %figure(1); visual(Vorg(:,index_ext),3,7);
    figure(1);
    index = randperm(2429);
    index_ext = index(1:rank_layers(1));    
    plot_dictionnary(Vorg(:,index_ext), [], [7 7]);
    
    figure(2); visual(semiW1,3,7);
    figure(3); visual(semiW2,3,5);
    figure(4); visual(semiW3,3,4);    
            
    figure(5); visual(nsW1,3,7);
    figure(6); visual(nsW2,3,5);
    figure(7); visual(nsW3,3,4);    
        
end


