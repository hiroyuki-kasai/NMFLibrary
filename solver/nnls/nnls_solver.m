function [H, WtW, WtV, infos] = nnls_solver(V, W, in_options) 
%   
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 20, 2022
%
% Change log: 
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.alg_name = 'no_name';
    local_options.algo = 'hals';
    local_options.delta = 1e-6;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);

    % initialize
    W = full(W); 
    WtW = W' * W;
    WtV = W' * V;
    infos = [];

    % If no initial matrices are provided, H is initialized as follows: 
    if ~isfield(options,'init') || isempty(options.init)
        H = nnls_init_nmflibrary(V, W, WtW, WtV); 
    else
        H = options.init; 
    end   

    % switch solvers

    switch options.algo
        case 'hals'
            
            % perform main routine
            options.V = V;
            options.W = W;
            options.main_routine_mode = true;
            %options.max_epoch = 500;
            options.eps0 = 0; 
            options.eps = 1;
            eit1 = cputime;
            alpha = 2;
        
            % call HALSupdt
            [H, infos] = HALSupdt(H, WtW, WtV, eit1, alpha, options.delta, options);              

            if options.verbose > 2
                fprintf('nnls_solver [%d]: H = %.16e, WtW = %.16e, WtV = %.16e\n', length(infos.cost), norm(H), norm(WtW), norm(WtV));    
            end

        case 'fpgm' 

            options_fpgm.init = H;
            options_fpgm.inner_max_epoch = options.inner_max_epoch;

            % call HALSupdt
            [H, WtW, WtV] = nnls_fpgm(V, W, options_fpgm);  

        case 'anls_bpp' 

            % call HALSupdt
            H = nnlsm_blockpivot(WtW, WtV, 1, H);  

        case 'anls_asgroup' 

            % call HALSupdt
            [H, ~, ~] = nnlsm_activeset(WtW, WtV, 0, 1, H);              

        otherwise
    end

 

end
