%
% demonstration file for NMFLibrary.
%
% This file is ported from https://github.com/dakuang/symnmf.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 17, 2022
%

close all
clear 
clc


X = load('graph.data');
n = [60, 140, 100];

tic; [idx, iter, obj, H] = symnmf_cluster(X, 3); toc;
disp(['Number of iterations: ', num2str(iter)]);
disp(['Objective function value: ', num2str(obj)]);

color = 'grb';
point = '.xo';
figure;
hold on;
count = 0;
for i = 1 : length(n)
    plot(X(count+1:count+n(i), 1), X(count+1:count+n(i), 2), [color(i), point(i)], 'MarkerSize', 8); 
    count = count + n(i);
end
set(gca, 'fontsize', 16);
set(gca, 'linewidth', 2);
xlabel('x_1', 'fontsize', 16);
ylabel('x_2', 'fontsize', 16);
title('300 data points', 'fontsize', 16);
axis equal;
