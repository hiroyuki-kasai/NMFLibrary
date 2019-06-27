function test_symm()
%
% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
% This demonstrates Symm-ANLS algorithm and Symm-Newton algorithm.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Jun. 24, 2019

    clc;
    clear;
    close all;

    %% generate synthetic data of (mxn) matrix       
    m = 1000;
    V = rand(m, m);
    V = (V + V')/2;
    
    
    %% Initialize of rank to be factorized
    rank = 5;
    
    
    %% set options
    options.verbose = 2;
    options.calc_symmetry = true; 



    %% perform factroization
    % Symm-ANLS
    options.alpha = 0.6;
    [w_symm_anls, infos_symm_anls] = symm_anls(V, rank, options);
    % Symm-Newton
    [w_symm_newton, infos_symm_newton] = symm_newton(V, rank, options);
    % Symm-Hals
    options.lambda = 0.6;    
    [w_symm_halsacc, infos_symm_halsacc] = symm_halsacc(V, rank, options);  
    
    
    %% plot
    display_graph('epoch','cost', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});
    display_graph('time','cost', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});
    
    %% symmetry
    display_graph('epoch','symmetry', {'Symm-ANLS', 'Symm-Newton', 'Symm-HALS'}, ...
        {w_symm_anls, w_symm_newton, w_symm_halsacc}, {infos_symm_anls, infos_symm_newton, infos_symm_halsacc});      
    
end


