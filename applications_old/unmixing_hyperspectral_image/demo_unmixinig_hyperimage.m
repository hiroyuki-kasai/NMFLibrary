%
% demonstration of unmixinig of hyperspectral image.
%
% This file has been ported from 
%       ONMF_Urban.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 20, 2022.
%


clear
clc
close all

% load 
if 0
    load('./Urban.mat'); 
    r = 6;
    display_rows = 2;
    display_cols = 3;    
elseif 0
    load('./F1_S_9.mat');
    [a, b, c] = size(S);
    X = zeros(a*b, c);
    for i = 1 : c
        mat = S(:,:,i);
        X(:, i) = mat(:);
    end
    X = X';
    r = 3;
    display_rows = 1;
    display_cols = r;
elseif 0
    load('./Salinas_hyperspectral.mat');
    [a, b, c] = size(Salinas_Image);
    X = zeros(a*b, c);
    for i = 1 : c
        mat = Salinas_Image(:,:,i);
        X(:, i) = mat(:);
    end
    X = X';
    r = 6;
    display_rows = 2;
    display_cols = 3;   
elseif 1
    load('./IndianPines_Data.mat');
    [a, b, c] = size(z);
    X = zeros(a*b, c);
    for i = 1 : c
        mat = z(:,:,i);
        X(:, i) = mat(:);
    end
    X = X';
    r = 6;
    display_rows = 2;
    display_cols = 3;      
end

[m, n] = size(X); 
 


% initialize with SPA
options.display = 0;
options.max_epoch = 100;
options.verbose = 1;    
spa_sol = spa(X, r, options); 


% perform NMF
options.x_init.W = X(:, spa_sol.K);
options.orth_h    = 1;
options.norm_h    = 2;
options.orth_w    = 0;
options.norm_w    = 0;    
[sol, infos] = orth_mu_nmf(X, r, options); 


%% plots
% plot cost
display_graph('epoch', 'cost', {'Orth-MU'}, {sol}, {infos});

% Display extracted spectral signatures, that is, columns of W
figure; 
plot(sol.W); 
title('extracted spectral signatures'); 

% Display results
plot_options.black_display = false;
plot_dictionnary(sol.H', [], [display_rows display_cols], plot_options);



