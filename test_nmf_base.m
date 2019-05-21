% Test file for NMF base 
%
% Created by H.Kasai on March 24, 2017
% Modified by H.Kasai on Oct. 18, 2017
% Modified by H.Kasai on May 21, 2019

clc;
clear;
close all;


anls_asgroup_flag = 1;
nmf_mu_flag = 1;
nmf_mod_mu_flag = 1;
nmf_acc_mu_flag = 1;
nmf_als_flag = 1;
nmf_hals_flag = 1;
nmf_acc_hals_flag = 1;
nmf_pgd_flag = 1;
nmf_direct_pgd_flag = 1;
nenmf_flag = 0;


%% load CBCL face datasets
fprintf('Loading data ...');
V = importdata('./data/CBCL_face.mat');
F = size(V, 1);
N = size(V, 2);
%N = 2000;
%V = V(:,1:N);
rank = 25;
Vo = V;
dim = N * F;
fprintf(' done\n');


%% set options
max_epoch = 100;
verbose = 1;


%% set initial data
x_init.W = rand(F, rank); 
x_init.H = rand(rank, N);
x_init.R = rand(F, N);


%% calculate optimal solution
calc_sol = 1;
if calc_sol
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
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

if nmf_mu_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;   
    options.alg = 'mu';
    
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
    
    names{alg_idx} = 'NMF MU'; 
    sols{alg_idx} = w_nmf_mu;
    infos{alg_idx} = infos_nmf_mu;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_mu.W, w_nmf_mu.H, zeros(F, N)) * 2 / dim;
end
    
if nmf_mod_mu_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'mu_mod';
    
    [w_mod_nmf_mu, infos_mod_nmf_mu] = nmf_mu(V, rank, options);
    
    names{alg_idx} = 'NMF Modified MU'; 
    sols{alg_idx} = w_mod_nmf_mu;
    infos{alg_idx} = infos_mod_nmf_mu;     
    costs{alg_idx} = nmf_cost(Vo, w_mod_nmf_mu.W, w_mod_nmf_mu.H, zeros(F, N)) * 2 / dim;    
end

if nmf_acc_mu_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'mu_acc';
    options.alpha = 2;
    options.delta = 0.1;
    
    [w_nmf_acc_mu, infos_nmf_acc_mu] = nmf_mu(V, rank, options);
    
    names{alg_idx} = 'NMF ACC MU'; 
    sols{alg_idx} = w_nmf_acc_mu;
    infos{alg_idx} = infos_nmf_acc_mu;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_acc_mu.W, w_nmf_acc_mu.H, zeros(F, N)) * 2 / dim;
end

if nmf_als_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose;
    options.f_opt = f_opt;       
    options.alg = 'als';
    
    [w_nmf_als, infos_nmf_als] = nmf_als(V, rank, options);
    
    names{alg_idx} = 'NMF ALS'; 
    sols{alg_idx} = w_nmf_als;
    infos{alg_idx} = infos_nmf_als;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_als.W, w_nmf_als.H, zeros(F, N)) * 2 / dim;
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

if nmf_acc_hals_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'acc_hals';
    
    [w_nmf_acc_hals, infos_nmf_acc_hals] = nmf_als(V, rank, options);
    
    names{alg_idx} = 'NMF ACC HALS'; 
    sols{alg_idx} = w_nmf_acc_hals;
    infos{alg_idx} = infos_nmf_acc_hals;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_acc_hals.W, w_nmf_acc_hals.H, zeros(F, N)) * 2 / dim;
end

if nmf_pgd_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'pgd';
    
    [w_nmf_pgd, infos_nmf_pgd] = nmf_pgd(V, rank, options);
    
    names{alg_idx} = 'NMF PGD'; 
    sols{alg_idx} = w_nmf_pgd;
    infos{alg_idx} = infos_nmf_pgd;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_pgd.W, w_nmf_pgd.H, zeros(F, N)) * 2 / dim;
end

if nmf_direct_pgd_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'direct_pgd';
    
    [w_nmf_direct_pgd, infos_nmf_direct_pgd] = nmf_pgd(V, rank, options);
    
    names{alg_idx} = 'NMF Direct PGD'; 
    sols{alg_idx} = w_nmf_direct_pgd;
    infos{alg_idx} = infos_nmf_direct_pgd;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_direct_pgd.W, w_nmf_direct_pgd.H, zeros(F, N)) * 2 / dim;
end

if nenmf_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'nenmf';
    
    [w_nenmf.W, w_nenmf.H, iter, elapse, HIS] = NeNMF(V, rank, 'max_iter', max_epoch, 'verbose', 2, 'w_init', x_init.W, 'h_init', x_init.H);
    
    names{alg_idx} = 'NeNMF (Nesterov Acc.)'; 
    sols{alg_idx} = w_nenmf;
    infos_nenmf.epoch = 0:max_epoch;
    infos_nenmf.cost = HIS.objf;    
    infos_nenmf.time = HIS.cpus;
    
    infos{alg_idx} = infos_nenmf;     
    costs{alg_idx} = nmf_cost(Vo, w_nenmf.W, w_nenmf.H, zeros(F, N)) * 2 / dim;
end

if anls_asgroup_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.f_opt = f_opt;       
    options.alg = 'anls_asgroup';
    options.alg = 'anls_asgivens';
    options.alg = 'anls_bpp';
    
    [w_anls_asgroup, infos_anls_asgroup] = nmf_anls(V, rank, options);
    
    names{alg_idx} = 'ANLS (Active Group)'; 
    sols{alg_idx} = w_anls_asgroup;
    infos{alg_idx} = infos_anls_asgroup;     
    costs{alg_idx} = nmf_cost(Vo, w_anls_asgroup.W, w_anls_asgroup.H, zeros(F, N)) * 2 / dim;
end


%% plot
display_graph('epoch','cost', names, sols, infos);
display_graph('time','cost', names, sols, infos);
display_graph('epoch','optimality_gap', names, sols, infos);
display_graph('time','optimality_gap', names, sols, infos);
