%
% demonstration of analysis of music data.
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

%rng('default')

% laod data

data_name = 'audio.wav';
%data_name = 'Piano.wav';

% obtain the matrix from an audio file, and convert audio file to matrix using TFCT
[X, Fs] = audioread(data_name); % X = data | Fs = sampling rate

% play audio
sound(X, Fs);

%% show (1)
figure

subplot(3,2,1)
plot(X);
title('Visualization of original audio data');
xlabel('temps');
ylabel('amplitude');

subplot(3,2,2)
spectrogram(X);
title('Spectrogram of original audio data');


%% perform NMF
X = X(1:100:end, 1:100:end); % sampling
%[TFR, T, F] = tfrstft(X); % TFCT
[TFR, F, T] = stft(X, Fs);
X = abs(TFR);

options.verbose = 1;
options.not_store_infos = true;
%options.alg = 'mu';
%[sol, infos] = fro_mu_nmf(X, 2, options);
options.alg = 'acc_hals';
[sol, ~] = als_nmf(X, 2, options); 
W = sol.W;
H = sol.H;


%% show (2)
subplot(3,2,3)
plot(H(1,:))
title('First line of H')
xlabel('temps')
ylabel('H(1,:)')

subplot(3,2,4)
plot(H(2,:))
title('Second line of H')
xlabel('temps')
ylabel('H(2,:)')

subplot(3,2,5)
plot(W(:,1))
title('First column of W')
xlabel('frequency')
ylabel('W(:,1)')

subplot(3,2,6)
plot(W(:,2))
title('DSecond column of W')
xlabel('frequency')
ylabel('W(:,2)')