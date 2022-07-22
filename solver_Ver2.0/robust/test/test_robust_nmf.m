function test_robust_nmf(varargin)
% Test file for NMF with outlier algorithm.
%
% Created by H.Kasai on May 21, 2019

    if nargin < 1
        clc;
        clear;
        close all;
        rng('default')


        %% generate/load data 
        % d=1: synthetic data in paper, 2: CBCL, 3: ORL, 4: UMISTface
        dataset = 'CBCL';
        % set the density of outlier
        rho = 0.5;
    
        fprintf('Loading data ...');
        [N, F, rank, Vo, V, Ro] = load_dataset(dataset, '../../../data', rho);    
        fprintf('done\n');

        [m, n] = size(V);
        options = [];
        options.verbose = 2;
        options.max_epoch = 1000;

        health_check_mode = false;
    else
        V = varargin{1};
        rank = varargin{2}; 
        options = varargin{3};
        health_check_mode = true;
    end

    [w_robust_mu, info_robust_mu] = robust_mu_nmf(V, rank, options);


    if ~health_check_mode
        [w_mu, info_mu] = fro_mu_nmf(V, rank, options); 

        %% plot
        display_graph('grad_calc_count','cost', {'MU', 'Robust-MU'}, {w_mu, w_robust_mu}, {info_mu, info_robust_mu});
    
        %% display basis elements obtained with different algorithms
        plot_dictionnary(w_mu.W, [], [7 7]); 
        plot_dictionnary(w_robust_mu.W, [], [7 7]);  
    end
end

