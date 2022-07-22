%
% demonstration of analysis of image data.
%
% This file has been ported from https://github.com/Fatiine/NMF-applied-to-music-.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 20, 2022.
%

close all
clear
clc

% load image
I = imread('lena.jpg');

% convert to shade of gray
I= get_luminance(I);
I = double(I);

% perform NMF for decomposition
options.verbose = 1;
options.not_store_infos = true;
%options.alg = 'mu';
%[sol, infos] = fro_mu_nmf(I, 100, options);
options.alg = 'acc_hals';
[sol, ~] = als_nmf(I, 100, options); 
W = sol.W;
H = sol.H;

%% show 
figure
subplot(2,2,1)
% show starting image
imagesc(I);
title('Original image');
colormap(gray);

% display result images
subplot(2,2,2)
imagesc(W);
title('W');
colormap(gray);

subplot(2,2,3)
imagesc(H);
title('H');
colormap(gray);

subplot(2,2,4)
imagesc(W*H);
title('W*H')
colormap(gray);


