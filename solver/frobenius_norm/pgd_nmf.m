function [x, infos] = pgd_nmf(V, rank, in_options)
% Projected gradient descent for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%       min || V - WH ||_F^2,
%       where 
%       {V, W, H} > 0.
%
% Given a non-negative matrix V, factorized non-negative matrices {W, H} are calculated.
%
%
% Inputs:
%       V           : (m x n) non-negative matrix to factorize
%       rank        : rank
%       in_options 
%           alg     : pgd: Projected gradient descent
%
%                   : fast_pgd: fast projected gradient descent with Nesterov' acceleration 
%
%                   : adp_step_pgd: Projected gradient descent with adaptive stepsize selection
%
%                   : direct_pgd: Projected gradient descent
%                       Reference:
%                           C.-J. Lin. 
%                           "Projected gradient methods for non-negative matrix factorization," 
%                           Neural Computation, vol. 19, pp.2756-2779, 2007.
%                           See https://www.csie.ntu.edu.tw/~cjlin/nmf/.
%                           The corresponding code is originally created by the authors, 
%                           This file is modifided by H.Kasai.
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.W: (m x rank), x.H: (rank x n)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
%
% This file is part of NMFLibrary
%
% Created by H.Kasai on Mar. 24, 2017
%
% Change log: 
%
%       Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%       Apr. 22, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%       May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%
%       Jun. 21, 2022 (Hiroyuki Kasai): Added fast pgd module.
%
%       Jun. 22, 2022 (Hiroyuki Kasai): Added momentum acceleration mode and mofified.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.alg = 'pgd';
    local_options.sub_mode = 'std'; 
    local_options.delta = 1e-6;    
    local_options.alpha = 1;
    local_options.tol_grad_ratio = 0.00001;
    local_options.inner_nnls_alg = 'hals';
    local_options.inner_max_epoch = 500;
    local_options.inner_max_epoch_parameter = 0.5;       
    local_options.beta0 = 0.5;
    local_options.eta = 1.5; 
    local_options.gammabeta = 1.01;
    local_options.gammabetabar = 1.005; 
    local_options.momentum_h = 0; 
    local_options.momentum_w = 0; 
    local_options.scaling = true;
    local_options.warm_restart = false;
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);      
    

    if ~strcmp(options.alg, 'pgd') && ~strcmp(options.alg, 'adp_step_pgd') && ~strcmp(options.alg, 'direct_pgd') && ~strcmp(options.alg, 'fast_pgd')
        fprintf('Invalid algorithm: %s. Therfore, we use pgd (i.e., projected gradient descent).\n', options.alg);
        options.alg = 'pgd';
    else
        options.alg = options.alg;
    end   
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
        
    % initialize
    method_name = sprintf('PGD (%s:%s)', options.alg, options.sub_mode);    
    epoch = 0;    
    grad_calc_count = 0; 
    stop_options = [];

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end  
    
    % intialize for pgd
    %tol_grad_ratio = options.tol_grad_ratio;
    if strcmp(options.alg, 'pgd')
        % do nothing
    elseif strcmp(options.alg, 'adp_step_pgd')
         if ~isfield(options, 'tol_grad_ratio')
            options.tol_grad_ratio = 0.00001; % tol = [0.001; 0.0001; 0.00001];
         end
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;   
        init_grad = norm([gradW; gradH'],'fro');
        tolW = max(0.001, options.tol_grad_ratio) * init_grad; 
        tolH = tolW;
        options.special_stop_condition = @(epoch, infos, options, stop_options) adp_step_pgd_stop_func(epoch, infos, options, stop_options);
    elseif strcmp(options.alg, 'direct_pgd')
         if ~isfield(options, 'tol_grad_ratio')
            options.tol_grad_ratio = 0.00001; % tol = [0.001; 0.0001; 0.00001];
         end        
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;           
        init_grad = norm([gradW; gradH'],'fro');    
        %H = nlssubprob(V, W, H, 0.001, options);    
        obj = nmf_cost(V, W, H, []);
        alpha = options.alpha;
    elseif strcmp(options.alg, 'fast_pgd')
        options_fpgm = [];
        options_fpgm.inner_max_epoch = options.inner_max_epoch;       
    end  
    
    if options.scaling
        [W, H] = normalize_WH(V, W, H, rank, 'type1');
    end
    
    [options, beta, betamax] = check_momemtum_setting(options);    
    
    if options.warm_restart
        nV = norm(V, 'fro');
        rel_error = zeros(1, options.max_epoch);
        rel_error(1) = sqrt(nV^2 - 2*sum(sum(V * H' .* W)) + sum(sum( H * H' .* (W'*W)))) / nV;          
    end
    W_prev = W; 
    H_prev = H;
    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('PGD (%s:%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, options.sub_mode, f_val, optgap); 
    end  
    
    % set start time
    start_time = tic();

    % main loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options, stop_options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end        
              
        if strcmp(options.alg, 'pgd') || strcmp(options.alg, 'fast_pgd') || strcmp(options.alg, 'adp_step_pgd')
          
            %% update H
            if strcmp(options.alg, 'pgd')            
                WtW = W' * W;
                L = norm(WtW, 'fro');  
                H = H - (WtW * H - W'*V) / L;

             elseif strcmp(options.alg, 'fast_pgd') 

                options_fpgm.init = H;
                options_fpgm.inner_max_epoch = change_inner_max_epoch(V, W, options);
                H = nnls_fpgm(V, W, options_fpgm); 

            elseif strcmp(options.alg, 'adp_step_pgd')

                % stopping condition
                projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
                stop_options.projnorm = projnorm;
                stop_options.init_grad = init_grad;

                %[H, gradH, iterH] = nlssubprob(V, W, H, tolH, options);
                nnls_options.init = H; 
                nnls_options.verbose = 0;
                nnls_options.inner_max_epoch = options.inner_max_epoch;
                nnls_options.algo = options.inner_nnls_alg;
                [H, ~, ~, infos_nnls_solver] = nnls_solver(V, W, nnls_options); % BCD on the rows of H   
                iterH = length(infos_nnls_solver.iter);

                if iterH == 1
                    tolH = 0.1 * tolH; 
                end 
                
            end

            % perform momentum for H 
            if strcmp(options.sub_mode, 'momentum')
                [H, H_tmp1, H_tmp2] = do_momentum_h(H, H_prev, beta, epoch, options);
            end
         
            
            %% update W
            if strcmp(options.alg, 'pgd')            

                HHt = H * H';
                L = norm(HHt, 'fro');              
                W = W - (W * HHt - V*H') / L;  

            elseif strcmp(options.alg, 'fast_pgd') 

                options_fpgm.init = W'; 
                options_fpgm.inner_max_epoch = change_inner_max_epoch(V', H', options);             
                Wt = nnls_fpgm(V', H', options_fpgm); 
                W = Wt';
          
            elseif strcmp(options.alg, 'adp_step_pgd')


                %[H, gradH, iterH] = nlssubprob(V, W, H, tolH, options);
                %[W, gradW, iterW] = nlssubprob(V', H', W', tolW, options); 
                nnls_options.init = W'; 
                nnls_options.verbose = 0;
                nnls_options.inner_max_epoch = options.inner_max_epoch;
                nnls_options.algo = options.inner_nnls_alg;
                [Wt, ~, ~, infos_nnls_solver] = nnls_solver(V', H', nnls_options); % BCD on the rows of H   
                iterW = length(infos_nnls_solver.iter);

                W = Wt'; 
                %gradW = gradW';
                if iterW == 1
                    tolW = 0.1 * tolW;
                end
                
            end 
            
            % perform momentum for W 
            if strcmp(options.sub_mode, 'momentum')
                [W, H, W_tmp1] = do_momentum_w(W, W_prev, H, H_prev, H_tmp1, beta, epoch, options);
            end            

        elseif strcmp(options.alg, 'direct_pgd')

            gradW = W * (H * H') - V * H';
            gradH = (W' * W) * H - W' * V;

            projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);  
            if projnorm < options.tol_grad_ratio * init_grad
                fprintf('final grad norm %f\n', projnorm);
            else
                [W, H, obj, alpha] = update_direct_pgd(V, W, H, gradW, gradH, alpha, obj); 
            end

        end
        
        % perform warm_restart
        if options.warm_restart && strcmp(options.sub_mode, 'momentum')
            [W, H, W_prev, H_prev, rel_error, beta, betamax, options] = ...
                warm_restart(V, W, H, rank, W_prev, H_prev, W_tmp1, H_tmp1, H_tmp2, rel_error, beta, betamax, epoch, options);
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;

        % store info
        [infos, f_val, optgap] = store_nmf_info(V, W, H, [], options, infos, epoch, grad_calc_count, elapsed_time); 
        
        % display info
        display_info(method_name, epoch, infos, options);

    end
    
    x.W = W;
    x.H = H;
    
end

function [W, H, obj, alpha] = update_direct_pgd(V, W, H, gradW, gradH, alpha, obj) 

    Wn = max(W - alpha * gradW, 0);    
    Hn = max(H - alpha * gradH, 0);    
    newobj = 0.5 * (norm(V - Wn*  Hn, 'fro')^2);

    if newobj - obj > 0.01 * (sum(sum(gradW .* (Wn-W))) + sum(sum(gradH .* (Hn-H))))
        % decrease stepsize    
        while 1
            alpha = alpha/10;
            Wn = max(W - alpha * gradW, 0);    
            Hn = max(H - alpha * gradH, 0);    
            newobj = 0.5 * (norm(V - Wn * Hn, 'fro')^2);

            if newobj - obj <= 0.01 * (sum(sum(gradW .* (Wn - W))) + sum(sum(gradH.*(Hn-H))))
                W = Wn; 
                H = Hn;
                obj = newobj;
            break;

            end
        end
    else 
        % increase stepsize
        while 1
            Wp = Wn; 
            Hp = Hn; 
            objp = newobj;
            alpha = alpha*10;
            Wn = max(W - alpha * gradW,0);    
            Hn = max(H - alpha * gradH,0);    
            newobj = 0.5 * (norm(V - Wn * Hn, 'fro')^2);

            %if (newobj - obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
            %    sum(sum(gradH.*(Hn-H))))) | (Wn==Wp & Hn==Hp)
            if (newobj - obj > 0.01 * (sum(sum(gradW .* (Wn - W))) + sum(sum(gradH .* (Hn-H))))) ...
                    || (isequal(Wn, Wp) && isequal(Hn, Hp))               
                W = Wp; 
                H = Hp;
                obj = objp; 
                alpha = alpha/10;
                break;
            end
        end
    end 
end


function [stop_flag, reason, rev_info] = adp_step_pgd_stop_func(epoch, infos, options, stop_options)
    stop_flag = false;
    reason = [];
    rev_info = [];

    if stop_options.projnorm < options.tol_grad_ratio * stop_options.init_grad
        stop_flag = true;
        reason = sprintf('Gradient norm tolerance reached: gnorm = %.4e < tol_ratio = %.4e * init_grad = %.4e)\n', stop_options.projnorm, options.tol_grad_ratio, stop_options.init_grad);
    end

end