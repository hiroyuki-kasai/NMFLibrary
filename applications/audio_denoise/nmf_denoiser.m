function [output_file] = nmf_denoiser(input_file, params)
%
% This file is ported from https://github.com/niklub/NMFdenoiser
%   by Hiroyuki Kasai for NMFLibrary (June 17, 2022).
%
%
%NMFdenoiser is able to perform audio noise reduction by using several
%Non-negative Matrix Factorization (NMF)-based algorithms.

%For further details please refer to the following paper:
%N. Lyubimov, M. Kotov "Non-negative Matrix Factorization with Linear
%Constraints for Single-Channel Speech Enhancement"
%(http://arxiv.org/abs/1309.6047)

%================USAGE==============================
%output_file = NMFdenoiser(input_file [,params])

%================INPUT PARAMETERS====================
%input_file:                    path to file that should be denoised
%params:                        (optional) input parameters (see below)
%   params.output ('')          output file name (set '' to produce
%                               <input_file>_DENOISED.wav
%   params.nwin (1024)          STFT window size (1024)
%   params.type ('denseNMF')    NMF type to perform:
%                               - 'NMF'     -standart NMF (similar to )
%                               - 'linNMF'  -linear NMF
%                               - 'denseNMF'-dense NMF
%   params.noise ('')           path to file containing noise profile or
%                               empty string if noise profile should 
%                               be estimated from target file
%   params.beta (1)             beta-divergence used in NMF estimations
%   params.m    (32)            number of NMF atoms to estimate
%   params.f0range([80 400])    the range in Hz of expected F0
%   params.fstep (10)           frequency step in Hz
%   params.num_atoms_per_f0 (4) number of harmonic atoms per each F0
%   params.num_noise_atoms (16) number of noise atoms
%   params.max_harms (30)       number of harmonics per each F0
%   params.show_log (true)      show messages
%   params.alpha(.2)            parameter controls envelope smoothness
%   params.energyThr (-40)      logRMS threshold in dB used in energy-based VAD
%   params.speech_sparsity(.2)  parameter controls speech sparsity

    if nargin<2
        params = struct();
    end

    if ~isfield(params, 'verbose')
        params.verbose = 0;
    end
    
    %[audio, sr] = wavread(input_file);
    [audio, sr] = audioread(input_file); % HK
    ainfo = audioinfo(input_file); % HK
    nbits = ainfo.BitsPerSample; % HK
    
    params = initParams(params, sr);
    if params.verbose
        fprintf('Processing input file %s...\n', input_file);
    end
    Saudio = m_STFT(audio, sr, params);
    Yaudio = abs(Saudio);
    
    if isempty(params.noise)
        if params.verbose
            fprintf('Extracting noise profile from target file...\n');
        end
        if exist('vadsohn')==2
            vad = vadsohn(audio, sr);
        else
            if params.verbose
                fprintf('WARNING! voicebox is not installed. The results could be inaccurrate\n');
            end
            vad_params.threshold = params.energyThr;
            vad = energyVAD(audio, vad_params);
        end
        noise_profile = audio(vad==0);
        if isempty(noise_profile)
            if params.verbose
                fprintf('Unable to find noise profile with prespecified params.energyThr. Using first samples instead\n');
            end
            noise_profile = audio(1:round(numel(audio)/10));
        end
            
    else
        if params.verbose
            fprintf('Extracting noise profile from file %s...\n', params.noise);
        end
        if ~exist(params.noise, 'file')
            fprintf('ERROR! params.noise==input is specified, but params.noise_profile doesn''t exist\n');
        end
        %noise_profile = wavread(params.noise);
        noise_profile = audioread(params.noise); % HK
    end
    if params.verbose
        fprintf('Computing noise atoms...\n');
    end
    Ynoise = abs(m_STFT(noise_profile, sr, params));
    if params.m >= size(Ynoise,2)
        fprintf('WARNING! Noise profile is too small\n');
        params.m = size(Ynoise,2)-1;
    end

    % init
    [init_factors, ~] = generate_init_factors(Ynoise, params.m, []);    
  

    % main routine
    clear nr_nmf_oprions;
    options.verbose = params.verbose;  
    if strcmp(params.alg, 'simple')
        params.metric = 'beta-div'; 
        nr_nmf_oprions.metric = params.metric;
        params.max_iter = params.first_max_epoch;
        params.D0 = init_factors.W;
        params.X0 = init_factors.H;  
        params.d_beta = 1;
        params.d_alpha = 2;
        Dnoise = simplestNMF(Ynoise, params);

    elseif strcmp(params.alg, 'mu_euc') || strcmp(params.alg, 'mu_alpha') ...
            || strcmp(params.alg, 'mu_beta') || strcmp(params.alg, 'mu_kl')   

        if strcmp(params.alg, 'mu_euc')
            options.metric_type = 'euc';               
        elseif strcmp(params.alg, 'mu_kl')
            options.metric_type = 'kl-div';  
        elseif strcmp(params.alg, 'mu_alpha')
            options.metric_type = 'alpha-div';             
        elseif strcmp(params.alg, 'mu_beta')
            options.metric_type = 'beta-div';            
        else
        end

        nr_nmf_oprions.metric = options.metric_type;        
        options.verbose = params.verbose;        
        options.d_beta = 1;
        options.max_epoch = params.first_max_epoch;        
        rank = params.m;
        options.x_init.W = init_factors.W;
        options.x_init.H = init_factors.H;   
        options.norm_w = false;
       [w_nmf_mu, ~] = fro_mu_nmf(Ynoise, rank, options);        
        Dnoise = w_nmf_mu.W; 
    else
        fprintf('Invalid algorithm.\n');
        return;
    end        

    if params.verbose
        fprintf('Creating signal mask using NMF...\n');
    end


    nr_nmf_oprions.verbose = params.verbose;      
    nr_nmf_oprions.alg = params.alg;
    nr_nmf_oprions.second_max_epoch = params.second_max_epoch;
    nr_nmf_oprions.d_beta = 1;
    W = noise_reduction_NMF(Yaudio, Dnoise, nr_nmf_oprions);

    if params.verbose    
        fprintf('Doing denoising transformations...\n');
    end
    output_audio = m_InverseSTFT(W.*Saudio, params);
    
    if isempty(params.output)
        [pathstr, name] = fileparts(input_file);
        output_file = sprintf('%s/%s_DENOISED.wav', pathstr, name);
    else
        output_file = params.output;
    end

    if params.verbose    
        fprintf('Saving output file to %s...\n', output_file);
    end
    audiowrite(output_file, output_audio, sr, 'BitsPerSample', nbits);

end

function pout = initParams(p, sr)
    pout = p;
    pout.sr = sr;
    if ~isfield(pout, 'nwin')           pout.nwin = 1024; end
    if ~isfield(pout, 'window')         pout.window = hann(pout.nwin); end
    if ~isfield(pout, 'type')           pout.type = 'denseNMF'; end
    if ~isfield(pout, 'noise')          pout.noise = ''; end
    if ~isfield(pout, 'beta')           pout.beta = 1; end
    if ~isfield(pout, 'm')              pout.m = 32; end
    if ~isfield(pout, 'f0range')        pout.f0range = [80 400]; end
    if ~isfield(pout, 'fstep')          pout.fstep = 10; end
    if ~isfield(pout, 'num_atoms_per_f0') pout.num_atoms_per_f0 = 4; end
    if ~isfield(pout, 'num_noise_atoms') pout.num_noise_atoms = 16; end
    if ~isfield(pout, 'max_harms')      pout.max_harms = 30; end
    if ~isfield(pout, 'output')         pout.output = ''; end
    if ~isfield(pout, 'show_log')       pout.show_log = true; end
    if ~isfield(pout, 'energyThr')       pout.energyThr = -40; end
    if ~isfield(pout, 'alpha')          pout.alpha = .2; end
    if ~isfield(pout, 'speech_sparsity')          pout.speech_sparsity = .2; end
end


function [W] = noise_reduction_NMF(noisy_magnitude, noise_matrix, nr_nmf_oprions, signal_components)
%"Reduction of Non-stationary Noise for a Robotic Living Assistant using
%Sparse Non-negative Matrix Factorization"

    if nargin<4
        ns = size(noise_matrix, 2);
    else
        ns = signal_components;
    end
    rank = ns + size(noise_matrix, 2);
    D0 = [rand(size(noisy_magnitude,1), ns), noise_matrix];
    X0 = rand(size(D0, 2), size(noisy_magnitude, 2));    

    if strcmp(nr_nmf_oprions.alg, 'mu_euc') || strcmp(nr_nmf_oprions.alg, 'mu_alpha') ...
            || strcmp(nr_nmf_oprions.alg, 'mu_beta') || strcmp(nr_nmf_oprions.alg, 'mu_kl')

        if strcmp(nr_nmf_oprions.alg, 'mu_euc')
            options.metric_type = 'euc';               
        elseif strcmp(nr_nmf_oprions.alg, 'mu_kl')
            options.metric_type = 'kl-div';  
        elseif strcmp(nr_nmf_oprions.alg, 'mu_alpha_d')
            options.metric_type = 'alpha-div';            
        elseif strcmp(nr_nmf_oprions.alg, 'mu_beta_d')
            options.metric_type = 'beta-div';            
        else
        end   

        options.verbose = nr_nmf_oprions.verbose;            
        options.max_epoch = nr_nmf_oprions.second_max_epoch;
        options.d_beta = 1;
        options.x_init.W = D0;
        options.x_init.H = X0;
        options.norm_w = false;        
        options.updateW = 1:ns;
        [w_nmf_mu_beta, ~] = fro_mu_partial_nmf(noisy_magnitude, rank, options);        
        Dout = w_nmf_mu_beta.W;
        Xout = w_nmf_mu_beta.H;  
    else
        fprintf('Invalid algorithm.\n');
        return;
    end             
    
    %create Wiener filter
    Ys = Dout(:,1:ns)*Xout(1:ns,:);
    Yn = Dout(:,(ns+1):end)*Xout((ns+1):end,:);
    W = Ys ./ (Ys + Yn);
end


function S = m_STFT(audio, sr, params)
    if nargin<3 
        params = struct();
    end
    params = initParams_STFT(params, sr);
    frames = buffer(audio, params.nwin, params.nwin-params.nhop, 'nodelay');
    frames = frames .* repmat(params.window, 1, size(frames,2));
    nfft = 2^nextpow2(params.nwin) * params.npadtimes;
    S = fft(frames, nfft, 1);
    S = S(1:floor(nfft/2)+1,:);
    if params.winscale
        scale = 2/sum(params.window);
        S = S * scale;
        S(1, :) = S(1, :)/2;
        S(end,:) = S(end, :)/2;
    end
end


function pout = initParams_STFT(p, sr)
    pout = p;
    pout.sr = sr;
    if ~isfield(pout, 'nwin')       pout.nwin = 512;                    end
    if ~isfield(pout, 'window')     pout.window = hann(pout.nwin);      end
    if ~isfield(pout, 'nhop')       pout.nhop = round(pout.nwin/4);     end
    if ~isfield(pout, 'npadtimes')  pout.npadtimes = 1;                 end
    if ~isfield(pout, 'winscale')   pout.winscale = false;              end
    
    pout.window = pout.window(:);
end


function audio = m_InverseSTFT(S, params)
    if nargin < 2
        params = struct();
    end
    [nfft, nframes] = size(S);
    nfft = 2*(nfft-1);
    params = initParams_invSTFT(params, nfft);
    if params.winscale
        scale = 2/sum(params.window);
        S = S / scale;
        S(1, :) = S(1,:)*2;
        S(end,:) = S(end,:)*2;
    end
    S = [S; conj(S(floor((nfft+1)/2):-1:2,:))];
    frames = real(ifft(S,[],1));
    frames = frames(1:params.nwin, :);
    frames = frames .* repmat(params.window, 1, nframes);
    audio = 2/3*unbuffer(frames,params.nwin,params.nwin-params.nhop);
end

function pout = initParams_invSTFT(p, nfft)
    pout = p;
    if ~isfield(pout, 'nwin')       pout.nwin = nfft;                 end
    if ~isfield(pout, 'window')     pout.window = hann(pout.nwin);      end
    if ~isfield(pout, 'nhop')       pout.nhop = round(pout.nwin/4);     end
    if ~isfield(pout, 'winscale')   pout.winscale = false;              end
    
    pout.window = pout.window(:);
end


function y = unbuffer(x,w,o)
    y    = [];
    skip = w - o;
    N    = ceil(w/skip);
    L    = (size(x,2) - 1) * skip + size(x,1);
    if size(x,1)<skip*N
        x(skip*N,end) = 0; 
    end
    for i = 1:N
        t = reshape(x(:,i:N:end),1,[]);
        l = length(t);
        y(i,l+(i-1)*skip) = 0;
        y(i,[1:l]+(i-1)*skip) = t;
    end;
    y = sum(y,1);
    y = y(1:L);
end


function [ vad ] = energyVAD( audio, params )
    %M_ENERGYVAD calculates voice activity from input audio using simple energy
    %threshold-based algorithm
    %INPUT:
    %   audio:  input audio samples [1xN]
    %   vad:    (0,1)-label array [1xN] where 1 indicates voice activity per samples
    
    if nargin<2
        params = struct();
    end
    params = initParams_energyVAD(params);
    frames = buffer(audio, params.nwin);
    rms = std(frames);
    vad = zeros(size(rms));
    vad(20*log10(rms)>params.threshold)=1;
    vad = kron(vad, ones(1, params.nwin));
    vad = vad(1:numel(audio));
end
    

function pout = initParams_energyVAD(p)
    
    pout = p;
    if ~isfield(pout, 'nwin')       pout.nwin = 64;     end
    if ~isfield(pout, 'threshold')  pout.threshold = -40; end
end

