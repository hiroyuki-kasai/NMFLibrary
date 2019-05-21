% Test file for GNMF and SDNMF 
%
% Created by Deng Cai
% Modified by H.Kasai on July 21, 2018

clc;
clear;
close all;


dataset = 'COIL20';
%dataset = 'PIE';


%% load CBCL face datasets
fprintf('Loading %s dataset ... ', dataset);
if strcmp(dataset, 'COIL20')
    load('../data/COIL20.mat');	
    fea = AllSet.X';
    gnd = AllSet.y';    
    nClass = class_num;
elseif strcmp(dataset, 'PIE')
    load('../data/PIE_pose27.mat');	
    nClass = length(unique(gnd));    
end
fprintf('done\n');

% % make dataset smaller for quick test
% perm_idx = randperm(size(fea,1));
% fea = fea(perm_idx,:);
% gnd = gnd(perm_idx);
% N = 500;
% fea = fea(1:N,:);
% gnd = gnd(1:N);

% Normalize each data vector to have L2-norm equal to 1 
[fea, ~] = data_normalization(fea, [], 'std');



%% Clustering in the original space
rand('twister', 5489);
label = litekmeans(fea, nClass, 'Replicates', 20);
mi = MutualInfo(gnd, label);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
fprintf('### k-means:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);            



%% NMF
options = [];
options.kmeansInit = 0;
options.maxIter = 100;
options.nRepeat = 1;
options.alpha = 0;
%when alpha = 0, GNMF boils down to the ordinary NMF.
rand('twister', 5489);
[~,V] = GNMF(fea', nClass, [], options); %'

% Clustering in the NMF subspace
rand('twister', 5489);
label = litekmeans(V', nClass, 'Replicates', 20);
mi = MutualInfo(gnd, label);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
fprintf('### NMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);  



%% GNMF
options = [];
options.WeightMode = 'Binary';  
W = constructW(fea,options);
options.maxIter = 100;
options.nRepeat = 1;
options.alpha = 100;
rand('twister', 5489);
[~,V] = GNMF(fea', nClass, W, options); %'

% Clustering in the GNMF subspace
rand('twister', 5489);
label = litekmeans(V', nClass, 'Replicates', 20);
mi = MutualInfo(gnd, label);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
fprintf('### GNMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);   



%% SDNMF
options = [];
options.WeightMode = 'Binary';  
W = constructW(fea,options);
options.maxIter = 100;
options.lambda = 100;
options.gamma = 10;
options.alpha = 10;
rand('twister', 5489);
[~,V] = SDNMF(fea', nClass, W, options);

% Clustering in the SDNMF subspace
rand('twister', 5489);
label = litekmeans(V, nClass, 'Replicates', 20);
mi = MutualInfo(gnd, label);
purity = calc_purity(gnd, label);
nmi = calc_nmi(gnd, label);
fprintf('### SDNMF:\tNMI:%5.4f, Purity:%5.4f, MutualInfo:%5.4f\n', nmi, purity, mi);  
