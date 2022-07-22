%
% demonstration file for NMFLibrary.
%
% This file is ported from https://github.com/niklub/NMFdenoiser.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 17, 2022
%


close all
clear 
clc

%================INPUT FILES====================
noisy_speech    = 'test/speech+noise_44khz.wav';
noise           = 'test/noise_44khz.wav';

%================INPUT PARAMS==================
params.nwin     = 1024;
params.show_log = false;
params.noise    = noise;

%================PROCESSING====================
fprintf('Playing input file %s...\n', noisy_speech);

%[noisy_speech_signal,sr] = wavread(noisy_speech);
[noisy_speech_signal, sr] = audioread(noisy_speech);
ainfo = audioinfo(noisy_speech);
nbits = ainfo.BitsPerSample;

%sound(noisy_speech_signal,sr);

fprintf('Denoising...\n');
%params.alg = 'simple';
%params.alg = 'mu_kl';
params.alg = 'mu_beta';
%params.alg = 'mu_euc';
params.first_max_epoch = 100;
params.second_max_epoch = 25;
%output_file = NMFdenoiser(noisy_speech,params);
output_file = nmf_denoiser(noisy_speech,params);

fprintf('Playing output file %s...\n', output_file);
%[denoised_speech_signal,sr] = wavread(output_file);
[denoised_speech_signal, sr] = audioread(output_file);
sound(denoised_speech_signal,sr);