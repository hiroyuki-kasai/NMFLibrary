% Test file for Probablistic NMF algorithms
%
% Created by H.Kasai on May 20, 2019

clc;
clear;
close all;

pnmf_vb_flag = 1;
pnmf_vb_ard_flag = 1;
nmf_mu_flag = 1;
nmf_hals_flag = 1;


%% load CBCL face datasets
% fprintf('Loading data ...');
%V = importdata('../../data/CBCL_face.mat');
V = dlmread('../../data/R.txt');
F = size(V, 1);
N = size(V, 2);
%N = 2000;
%V = V(:,1:N);
rank = 10;
Vo = V;
dim = N * F;
fprintf(' done\n');


%% set options
max_epoch = 300;
verbose = 2;


%% set initial data
%x_init.W = ones(F, rank); 
%x_init.H = ones(rank, N);
%x_init.W = rand(F, rank); 
%x_init.H = rand(rank, N);
%x_init.R = zeros(F, N);
x_init = [];



%% calculate optimal solution
calc_sol = 1;
if calc_sol
    clear options;
    options.max_epoch = max_epoch;
    %options.x_init = x_init;
    options.verbose = 0;  
    options.max_epoch = 1000;
    
    if 1
        fprintf('Calculating f_opt by HALS ...\n');
        options.alg = 'hals';
        [w_sol, infos_sol] = nmf_als(V, rank, options);
        f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
        
    else
        fprintf('Calculating f_opt by ANLS ...\n');
        options.alg = 'anls_asgroup';
        options.alg = 'anls_asgivens';
        options.alg = 'anls_bpp';
        [w_sol, infos_sol] = nmf_anls(V, rank, options);
        f_opt = nmf_cost(V, w_sol.W, w_sol.H, zeros(F, N));
    end
    fprintf('Done.. f_opt: %.16e\n', f_opt);
else
    f_opt = -Inf;
end


%% execute algorithms
names = cell(1);
sols = cell(1);
infos = cell(1);
costs = cell(1);
alg_idx = 0;


if pnmf_vb_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    %options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;   
    
    %
    lambdaU = 0.1;
    lambdaV = 0.1;
    alphatau = 1;
    betatau = 1;
    alpha0 = 1;
    beta0 = 1;
    options.hyperparams.alphatau = alphatau;
    options.hyperparams.betatau = betatau;
    options.hyperparams.alpha0 = alpha0;
    options.hyperparams.beta0 = beta0;
    options.hyperparams.lambdaU = lambdaU;
    options.hyperparams.lambdaV = lambdaV;
    options.ard = false;    
    options.init_alg = 'prob_expectation';
    %options.init_alg = 'prob_random';
    
    [w_pnmf_vb, infos_pnmf_vb] = pnmf_vb(V, rank, options);
    
    names{alg_idx} = 'PNMF VB'; 
    sols{alg_idx} = w_pnmf_vb;
    infos{alg_idx} = infos_pnmf_vb;     
    costs{alg_idx} = nmf_cost(Vo, w_pnmf_vb.W, w_pnmf_vb.H, zeros(F, N)) * 2 / dim;
end

if pnmf_vb_ard_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    %options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;   
    
    %
    lambdaU = 0.1;
    lambdaV = 0.1;
    alphatau = 1;
    betatau = 1;
    alpha0 = 1;
    beta0 = 1;
    options.hyperparams.alphatau = alphatau;
    options.hyperparams.betatau = betatau;
    options.hyperparams.alpha0 = alpha0;
    options.hyperparams.beta0 = beta0;
    options.hyperparams.lambdaU = lambdaU;
    options.hyperparams.lambdaV = lambdaV;
    options.ard = true;    
    options.init_alg = 'prob_expectation';
    %options.init_alg = 'prob_random';
    
    [w_pnmf_vb, infos_pnmf_vb] = pnmf_vb(V, rank, options);
    
    names{alg_idx} = 'PNMF VB (ARD)'; 
    sols{alg_idx} = w_pnmf_vb;
    infos{alg_idx} = infos_pnmf_vb;     
    costs{alg_idx} = nmf_cost(Vo, w_pnmf_vb.W, w_pnmf_vb.H, zeros(F, N)) * 2 / dim;
end


if nmf_mu_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;   
    options.alg = 'mu';
    options.init_alg = 'NNDSVD';
    
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
    
    names{alg_idx} = 'NMF MU'; 
    sols{alg_idx} = w_nmf_mu;
    infos{alg_idx} = infos_nmf_mu;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_mu.W, w_nmf_mu.H, zeros(F, N)) * 2 / dim;
end
    

if nmf_hals_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'hals';
    
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);
    
    names{alg_idx} = 'NMF HALS'; 
    sols{alg_idx} = w_nmf_hals;
    infos{alg_idx} = infos_nmf_hals;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_hals.W, w_nmf_hals.H, zeros(F, N)) * 2 / dim;
end


%% plot
display_graph('epoch','cost', names, sols, infos);
display_graph('time','cost', names, sols, infos);
display_graph('epoch','optimality_gap', names, sols, infos);
display_graph('time','optimality_gap', names, sols, infos);


