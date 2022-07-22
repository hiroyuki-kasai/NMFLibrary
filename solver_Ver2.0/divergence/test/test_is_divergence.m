% demonstration file for NMFLibrary.
%
% This file illustrates how to use this library. 
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 24, 2017
%
% Change log: 
%
%   July 02, 2022 (Hiroyuki Kasai): Modified descriptions.
%
%

rng('default')

clc;
clear;
close all;

%% generate synthetic data of (mxn) matrix
m = 200;
n = 300;
V = rand(m,n);


%% Initialize of rank to be factorized
rank = 10;


%% Initialize of W and H
options = [];
[x_init.W, x_init.H] = NNDSVD(abs(V), rank, 0);
options.x_init = x_init;
options.verbose = 1;
options.max_epoch = 1000;
options.norm_w = true;


% 'beta-div';
options.metric_type = 'beta-div';
options.d_beta = 0;
[w_nmf_mu_beta0, infos_nmf_mu_beta0] = div_mu_nmf(V, rank, options);

options.metric_type = 'beta-div';
options.d_beta = 0;
options.rho = 50000;
[~, infos_nmf_is_admm50000] = div_admm_nmf(V, rank, options);

options.rho = 5000;
[~, infos_nmf_is_admm5000] = div_admm_nmf(V, rank, options);

options.rho = 1000;
[~, infos_nmf_is_admm1000] = div_admm_nmf(V, rank, options);

options.rho = 500;
[~, infos_nmf_is_admm500] = div_admm_nmf(V, rank, options);


%% plot
display_graph('epoch','cost',{'div-MU (beta=0:IS)','div-ADMM (beta=0:IS,rho=50000)','div-ADMM (beta=0:IS,rho=5000)','div-ADMM (beta=0:IS,rho=1000)','div-ADMM (beta=0:IS,rho=500)'}, ...
    [], {infos_nmf_mu_beta0, infos_nmf_is_admm50000, infos_nmf_is_admm5000, infos_nmf_is_admm1000, infos_nmf_is_admm500});

display_graph('time','cost',{'div-MU (beta=0:IS)','div-ADMM (beta=0:IS,rho=50000)','div-ADMM (beta=0:IS,rho=5000)','div-ADMM (beta=0:IS,rho=1000)','div-ADMM (beta=0:IS,rho=500)'}, ...
    [], {infos_nmf_mu_beta0, infos_nmf_is_admm50000, infos_nmf_is_admm5000, infos_nmf_is_admm1000, infos_nmf_is_admm500});
