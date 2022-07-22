function test_rank2nmf_in_non_rank2matrix()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates rank2nmf algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 13, 2022

    clc;
    clear;
    close all;
    
    %rng('default')    

    %% generate synthetic data of (mxn) matrix       
    m = 10;
    n = 200;
    options.verbose = 0;
    options.max_epoch = 100;
    
    %% rank-two matrix case
    % Initialize factor matrices for rank-two matrix
    V = rand(m,2)*rand(2,n); 
  
    % perform factroization
    options.alg = 'acc_hals';    
    [x_rank2nmf, infos_rank2nmf] = rank2nmf(V, options);        

    % display relative error
    fprintf('The relative error ||V-WH||_F / ||V||F is %2.2f%%.\n', ... 
        100 * norm(V - x_rank2nmf.W * x_rank2nmf.H, 'fro') / norm(V,'fro'))   


    %% non rank-two matrix case
    % Initialize factor matrices for rank-two matrix
    V = rand(m,2)*rand(2,n) + 0.05*rand(m,n); 
  
    % perform factroization
    options.alg = 'acc_hals';    
    [x_rank2nmf, infos_rank2nmf] = rank2nmf(V, options);        

    % display relative error
    fprintf('The relative error ||V-WH||_F / ||V||F is %2.2f%%.\n', ... 
        100 * norm(V - x_rank2nmf.W * x_rank2nmf.H, 'fro') / norm(V,'fro'))     
      
    
end


