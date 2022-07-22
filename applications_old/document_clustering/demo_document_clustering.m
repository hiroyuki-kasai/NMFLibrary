%
% demonstration file for NMFLibrary.
%
% This is a template script for document clustering problem using NMF. 
%
% This file is ported from
% https://github.com/abozied/Arabic-Documents-Clustering-Using-NMF.
%
% This file is part of NMFLibrary.
%
% Ported by H.Kasai on June 20, 2022
%


clc
clear
close all

% import datasets
%load('Reuters21578.mat');
%load('20NewsHome.mat');
load('TDT2.mat');
%load('PIE_pose27.mat');

nClass = length(unique(gnd));
fea = NormalizeFea(fea);
%[fea, ~] = normalize_data(fea, gnd, 'mean_std');
%[fea2, ~] = normalize_data(fea, gnd, 'std');
%[fea3, ~] = normalize_data(fea, gnd, 'mean');
fea = fea';



%% Clustering in the original feature space
label = litekmeans(fea',nClass);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
label = best_map(gnd,label);
MIhat = MutualInfo(gnd,label);
AC = length(find(gnd == label))/length(gnd);
fprintf('## Original domain: purity = %.4f, nmi = %.4f, mutual_info = %.4f, accuracy = %.4f\n', purity, nmi, MIhat, AC);



%% NMF with ACC-HALS
clear options
options.max_epoch = 100;
options.verbose = 1;
options.not_store_infos = true;
%options.alg = 'mu';
%[sol, infos] = nmf_mu(fea, nClass, options);
options.alg = 'acc_hals';
[sol, ~] = als_nmf(fea, nClass, options); 
V = sol.H;

label = litekmeans(V',nClass);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
label_permute = best_map(gnd,label);
MIhat = MutualInfo(gnd,label_permute);
AC = length(find(gnd == label_permute))/length(gnd);
fprintf('## NMF (ACC-HALS): purity = %.4f, nmi = %.4f, mutual_info = %.4f, accuracy = %.4f\n', purity, nmi, MIhat, AC);



%% NMF with Orth-MU
clear options
options.max_epoch = 100;
options.verbose = 1;
options.not_store_infos = true;
options.orth_h    = 1;
options.norm_h    = 2;
options.orth_w    = 0;
options.norm_w    = 0;    
[sol, ~] = orth_mu_nmf(fea, nClass, options); 
V = sol.H;

label = litekmeans(V',nClass);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
label_permute = best_map(gnd,label);
MIhat = MutualInfo(gnd,label_permute);
AC = length(find(gnd == label_permute))/length(gnd);
fprintf('## NMF (Orth-MU): purity = %.4f, nmi = %.4f, mutual_info = %.4f, accuracy = %.4f\n', purity, nmi, MIhat, AC);
