%
% demonstration file for NMFLibrary.
%
% This is a template script for signal separation problem using NMF. 
%
% This file is ported from https://github.com/kaankurtca/SignalSeparation_usingNMF.
%
% This file is part of NMFLibrary.
%
% Ported by H.Kasai on June 18, 2022
%


clc; 
clear
close all;

% set 
save_file_flag = false;
play_mixed_signal = true;

% load wav files
[music,Fs1] = audioread('musicf1.wav');
%sound(music,Fs1);
[speech,Fs2] = audioread('speechf1.wav');
%sound(speech,Fs2);
[mixed,Fs3] = audioread('mixedf1.wav');
%sound(mixed,Fs3); 

% compute mag. spectrograms of music and speech
music_spectrum = stft(music', 2048, 256, 0, hann(2048));
MagMusic = abs(music_spectrum);
PhaseMusic = music_spectrum./MagMusic;

speech_spectrum = stft(speech', 2048, 256, 0, hann(2048));
MagSpeech = abs(speech_spectrum);
PhaseSpeech = speech_spectrum./MagSpeech;     

% compute mag. spectrograms of mixed signals
mixed_spectrum = stft(mixed', 2048, 256, 0, hann(2048));
MagMixed = abs(mixed_spectrum);
PhaseMixed = mixed_spectrum./MagMixed;    % Magnitude spectrograms and Phases are computed seperately.


% load initial base and weight matrices
Bminit = load('Bminit.mat', 'Bm'); Bminit=Bminit.Bm;
Wminit = load('Wminit.mat', 'Wm'); Wminit=Wminit.Wm;
Bsinit = load('Bsinit.mat', 'Bs'); Bsinit=Bsinit.Bs;
Wsinit = load('Wsinit.mat', 'Ws'); Wsinit=Wsinit.Ws; 



% perform factroization
%options.alg = 'mu';
options.alg = 'mu_acc';
max_epoch = 250;
options.max_epoch = 250;
options.verbose = 1;

fprintf('## Extracting basis of music ...\n');
[sol_music, infos_music] = fro_mu_nmf(MagMusic, size(Bminit,2), options);
Bm = sol_music.W;
Wm = sol_music.H;

fprintf('## Extracting basis of speech ...\n');
[sol_speech, infos_speech] = fro_mu_nmf(MagSpeech, size(Bsinit,2), options);
Bs = sol_speech.W;
Bw = sol_speech.H;    

if save_file_flag
    save("BasesForMusic(NMF).mat", "Bm"); 
    save("BasesForSpeech(NMF).mat", "Bs");
end

% separate signals 
fprintf('## Separating mixed signal into music and speech signals ...\n');
[music_recv, speech_recv] = separate_signals(MagMixed, Bm, Bs, max_epoch);
 
if play_mixed_signal
    fprintf('## Playing original mixed signal (music + speech) ...\n');
    sound(mixed, Fs3); 
    pause(20)   %so that the voices do not interfere.
end

% multiply by phase and reconstruct time domain signal by inverse stft.
fprintf('## Playing separated music ...\n');
seperatedMusic = stft((music_recv.*PhaseMixed),2048,256,0,hann(2048));
seperatedMusic = transpose(seperatedMusic);
sound(seperatedMusic,Fs3);

pause(20)   %so that the voices do not interfere.
  
fprintf('## Playing separated speech ...\n');
seperatedSpeech = stft((speech_recv.*PhaseMixed),2048,256,0,hann(2048));
seperatedSpeech = transpose(seperatedSpeech);
sound(seperatedSpeech,Fs3);


if save_file_flag
    % write time domain speech and music with 16000 sampling
    Fs = 16000;
    audiowrite('seperatedMusic.wav', seperatedMusic, Fs);
    udiowrite('seperatedSpeech.wav', seperatedSpeech, Fs); % Finally, they are saved as waw files with 16k samples.
end

pause(20)   %so that the voices do not interfere.

fprintf('## Done.\n');