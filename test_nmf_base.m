% Test file for NMF base 
%
% Created by H.Kasai on March. 24, 2017

clc;
clear;
close all;

%rng(12345)

anls_asgroup_flag = 1;
nmf_mu_flag = 1;
nmf_mod_mu_flag = 1;
nmf_acc_mu_flag = 1;
%nmf_acc_mu_new_flag = 1;
nmf_als_flag = 1;
nmf_hals_flag = 1;
nmf_acc_hals_flag = 1;
nmf_pgd_flag = 1;
nmf_direct_pgd_flag = 1;
nenmf_flag = 0;

%% generate/load data 
% d=1: synthetic data in paper, 2: CBCL, 3: ORL, 4: UMISTface
d = 2;
% set the density of outlier
rho = 0.0;

fprintf('Loading data ...');
[N, F, K, Vo, V, Ro] = load_dataset(d, rho);
fprintf('done\n');
dim = N * F;


%% set options
max_epoch = 100;
verbose = 1;


%% set initial data
x_init.W = rand(F, K); 
x_init.H = rand(K, N);
x_init.R = rand(F, N);



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
    options.alg = 'mu';
    
    [w_nmf_mu, infos_nmf_mu] = nmf_mu(V, K, options);
    
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
    options.alg = 'mod_mu';
    
    [w_mod_nmf_mu, infos_mod_nmf_mu] = nmf_mu(V, K, options);
    
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
    options.alg = 'acc_mu';
    options.alpha = 2;
    options.delta = 0.1;
    
    [w_nmf_acc_mu, infos_nmf_acc_mu] = nmf_mu(V, K, options);
    
    names{alg_idx} = 'NMF ACC MU'; 
    sols{alg_idx} = w_nmf_acc_mu;
    infos{alg_idx} = infos_nmf_acc_mu;     
    costs{alg_idx} = nmf_cost(Vo, w_nmf_acc_mu.W, w_nmf_acc_mu.H, zeros(F, N)) * 2 / dim;
end

% if nmf_acc_mu_new_flag
%     alg_idx = alg_idx + 1;  
%     clear options;
%     options.max_epoch = max_epoch;
%     options.x_init = x_init;
%     options.verbose = verbose; 
%     options.alg = 'acc_mu_new';
%     options.alpha = 2;
%     options.delta = 0.1;
%     
%     [w_nmf_acc_mu_new, infos_nmf_acc_mu_new] = nmf(V, K, options);
%     
%     names{alg_idx} = 'NMF ACC MU New'; 
%     sols{alg_idx} = w_nmf_acc_mu_new;
%     infos{alg_idx} = infos_nmf_acc_mu_new;     
%     costs{alg_idx} = nmf_cost(Vo, w_nmf_acc_mu_new.W, w_nmf_acc_mu_new.H, zeros(F, N)) * 2 / dim;
% end

if nmf_als_flag
    alg_idx = alg_idx + 1;  
    clear options;
    options.max_epoch = max_epoch;
    options.x_init = x_init;
    options.verbose = verbose; 
    options.alg = 'als';
    
    [w_nmf_als, infos_nmf_als] = nmf_als(V, K, options);
    
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
    options.alg = 'hals';
    
    [w_nmf_hals, infos_nmf_hals] = nmf_als(V, K, options);
    
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
    options.alg = 'acc_hals';
    
    [w_nmf_acc_hals, infos_nmf_acc_hals] = nmf_als(V, K, options);
    
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
    options.alg = 'pgd';
    
    [w_nmf_pgd, infos_nmf_pgd] = nmf_pgd(V, K, options);
    
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
    options.alg = 'direct_pgd';
    
    [w_nmf_direct_pgd, infos_nmf_direct_pgd] = nmf_pgd(V, K, options);
    
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
    options.alg = 'nenmf';
    
    [w_nenmf.W, w_nenmf.H, iter, elapse, HIS] = NeNMF(V, K, 'max_iter', max_epoch, 'verbose', 2, 'w_init', x_init.W, 'h_init', x_init.H);
    
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
    options.alg = 'anls_asgroup';
    options.alg = 'anls_asgivens';
    options.alg = 'anls_bpp';
    
    [w_anls_asgroup, infos_anls_asgroup] = nmf_anls(V, K, options);
    
    names{alg_idx} = 'ANLS (Active Group)'; 
    sols{alg_idx} = w_anls_asgroup;
    infos{alg_idx} = infos_anls_asgroup;     
    costs{alg_idx} = nmf_cost(Vo, w_anls_asgroup.W, w_anls_asgroup.H, zeros(F, N)) * 2 / dim;
end







%% plot
display_graph('epoch','cost', names, sols, infos);
display_graph('time','cost', names, sols, infos);
%display_graph('epoch','optimality_gap', names, sols, infos);


alg_total = alg_idx;
for alg_idx=1:alg_total
    fprintf('%s: MSE:%e, time:%e [sec]\n', names{alg_idx}, costs{alg_idx}, infos{alg_idx}.time(end));
    if d == 4
        %figure;
        %plot_dictionnary(sols{alg_idx}.W, [], [7 7]); 
    end
end


