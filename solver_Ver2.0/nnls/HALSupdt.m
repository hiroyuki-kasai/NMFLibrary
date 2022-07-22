function [V, infos] = HALSupdt(V, UtU, UtM, eit1, alpha, delta, in_options)
% Computes an approximate solution of the following nonnegative least
% squares problem (NNLS).
%
% The problem of interest is defined as
%
%       min || X - WV ||_F^2,
%       where 
%       {V} >= 0.
%
%       which is solved by an exact block-coordinate descent scheme.
%       This code is also used for updating step for accelerated HALS NMF.
%
% References:
%       N. Gillis and F. Glineur, 
%       "Accelerated Multiplicative Updates and hierarchical ALS Algorithms for Nonnegative 
%       Matrix Factorization,", 
%       Neural Computation 24 (4), pp. 1085-1105, 2012. 
%    
%
%   
% This file is part of NMFLibrary.
%
% Originally created by N. Gillis
%       See https://sites.google.com/site/nicolasgillis/publications.
%
%       This file has been ported from 
%       NNLS/nnls_HALSupdt.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)
%
% Change log: 
%
%       Jan. 01, 2018 (Hiroyuki Kasai):     Ported initial version
%
%       Jun. 08, 2022 (Hiroyuki Kasai):     Modified for use as a main routine mode
%

    [r, n] = size(V); 
    eit2 = cputime; % Use actual computational time instead of estimates rhoU
    cnt = 1; % Enter the loop at least once
    eps = 1; 
    eps0 = 1; 
    eit3 = 0;
    main_routine_mode = false;

    if nargin < 7
        options = [];
    else
        options = in_options;

        if isfield(options, 'main_routine_mode')
            main_routine_mode = options.main_routine_mode;
        else
            main_routine_mode = false;
        end          

        if isfield(options, 'eit2')
            eit2 = options.eit2;
        else
            eit2 = cputime;
        end  

        if isfield(options, 'eps')
            eps = options.eps;
        else
            eps = 1;
        end  

        if isfield(options, 'eps0')
            eps0 = options.eps0;
        else
            eps0 = 1;
        end

        if isfield(options, 'eit3')
            eit3 = options.eit3;
        else
            eit3 = 1;
        end     

        

    end

    if main_routine_mode 

        % initialize
        epoch = 0; 
        grad_calc_count = 0;
        
        % store initial info
        clear infos;
        [infos, f_val, optgap] = store_nmf_info(options.V, options.W, V, [], options, [], epoch, grad_calc_count, 0);
        
        if options.verbose > 2
            fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg_name, f_val, optgap); 
        end     
             
        % set start time
        start_time = tic();
    else
        infos = [];
    end    

    while cnt == 1 || (cputime-eit2 < (eit1+eit3)*alpha && eps >= (delta)^2*eps0)
        nodelta = 0; 
        if cnt == 1 
            eit3 = cputime; 
        end
            for k = 1 : r
                deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
                V(k,:) = V(k,:) + deltaV;
                nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
                if V(k,:) == 0
                    V(k,:) = 1e-16*max(V(:)); 
                end % safety procedure
            end
        if cnt == 1
            eps0 = nodelta; 
            eit3 = cputime-eit3; 
        end
        eps = nodelta; cnt = 0; 

        if main_routine_mode 
            grad_calc_count = grad_calc_count + r*n;
    
            % measure elapsed time
            elapsed_time = toc(start_time);        
    
            % update epoch
            epoch = epoch + 1;        
            
            % store info
            [infos, f_val, optgap] = store_nmf_info(options.V, options.W, V, [], options, infos, epoch, grad_calc_count, elapsed_time);          
            
            % display infos
            display_info(options.alg_name, epoch, infos, options);

            if epoch >= options.max_epoch
                break;
            end
        end

    end

    if main_routine_mode     

        if options.verbose > 0
            if optgap < options.tol_optgap
                fprintf('# %s: Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg_name, f_val, options.f_opt, options.tol_optgap);
            elseif epoch == options.max_epoch
                fprintf('%s: Max epoch reached (%g).\n', options.alg_name, options.max_epoch);
            end 
        end

    end
end