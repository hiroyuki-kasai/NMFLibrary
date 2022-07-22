function demo_denoiser_snr()
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
    noisy_speech    = 'test/speech+noise_8khz.wav';
    noise           = 'test/noise_8khz.wav';
    clean_speech    = 'test/speech_8khz.wav';
    
    %================OUTPUT FILES====================
    denoised_ss_simple_nmf     = 'test/denoised_8khz_SS_SimpleNMF.wav';
    denoised_ss_mu_euc_nmf     = 'test/denoised_8khz_SS_MuEucNMF.wav';
    denoised_ss_mu_alpha_nmf     = 'test/denoised_8khz_SS_MuAlphaNMF.wav';    
    denoised_ss_mu_beta_nmf     = 'test/denoised_8khz_SS_MuBetaNMF.wav';
    denoised_ss_mu_kl_nmf     = 'test/denoised_8khz_SS_MuKlNMF.wav';

    denoised_us_simple_nmf     = 'test/denoised_8khz_US_SimpleNMF.wav';
    denoised_us_mu_euc_nmf     = 'test/denoised_8khz_US_MuEucNMF.wav';
    denoised_us_mu_alpha_nmf     = 'test/denoised_8khz_US_MuAlphaNMF.wav';    
    denoised_us_mu_beta_nmf     = 'test/denoised_8khz_US_MuBetaNMF.wav';
    denoised_us_mu_kl_nmf     = 'test/denoised_8khz_US_MuKlNMF.wav';    

    %================INPUT PARAMS==================
    params.nwin = 256;
    params.show_log = false;
    rank = 32;
    
    %================PROCESSING====================
    fprintf('The input file %s has SNR = %.2f dB\n',...
        noisy_speech, getSNR(clean_speech, noisy_speech));
    
    %-------------SEMI-SUPERVISED---------------
    fprintf('\n================\n');
    fprintf('Semi-supervised (SS) denoisers...\n');
    params.noise    = noise;
    
    
    first_max_epoch = 100;
    second_max_epoch = 25;
    
    params.verbose = 0;
    
    
    rng('default')
    fprintf('Fro-MU (euc)...');
    params.output   = denoised_ss_mu_euc_nmf;
    params.alg = 'mu_euc';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));
    
    rng('default')
    fprintf('div-MU (beta-div)...');
    params.output   = denoised_ss_mu_beta_nmf;
    params.alg = 'mu_beta_d';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));
    
    rng('default')
    fprintf('div-MU (kl-div)...');

    params.output   = denoised_ss_mu_kl_nmf;
    params.alg = 'mu_kl';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));
    

    
    %-------------UNSUPERVISED---------------
    fprintf('\n================\n');
    fprintf('Unsupervised (US) denoisers...\n');
    params.noise = '';
    
    %----------NMF-------------
    rng('default')
    fprintf('Fro-MU (euc)...');
    params.output   = denoised_us_mu_euc_nmf;
    params.alg = 'mu_euc';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));

    rng('default')
    fprintf('div-NMF (beta-div)...');
    %params.type     = 'NMF';
    params.output   = denoised_us_mu_beta_nmf;
    params.alg = 'mu_beta_d';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));
    
    rng('default')
    fprintf('div-NMF (kl-div)...');
    %params.type     = 'NMF';
    params.output   = denoised_us_mu_kl_nmf;
    params.alg = 'mu_kl';
    params.first_max_epoch = first_max_epoch;
    params.second_max_epoch = second_max_epoch;
    nmf_denoiser(noisy_speech, params);
    fprintf(' Done! SNR = %.2f dB\n', getSNR(clean_speech, params.output));
    

end


function SNR = getSNR(clean_file, noisy_file)
    clean_audio = audioread(clean_file);
    noisy_audio = audioread(noisy_file);
    
    n = min(numel(clean_audio), numel(noisy_audio));
    SNR =  20*log10(norm(clean_audio(1:n))/norm(clean_audio(1:n)-noisy_audio(1:n)));
end