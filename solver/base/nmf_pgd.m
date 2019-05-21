function [x, infos] = nmf_pgd(V, rank, in_options)
% Projected gradient descent for non-negative matrix factorization (NMF).
%
% The problem of interest is defined as
%
%           min || V - WH ||_F^2,
%           where 
%           {V, W, H} > 0.
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
% Created by H.Kasai on Mar. 24, 2017
%
% Change log: 
%
%   Oct. 27, 2017 (Hiroyuki Kasai): Fixed algorithm. 
%
%   Apr. 22, 2019 (Hiroyuki Kasai): Fixed bugs.
%
%   May. 20, 2019 (Hiroyuki Kasai): Added initialization module.
%


    % set dimensions and samples
    [m, n] = size(V);
 
    % set local options
    local_options = []; 
    local_options.alg   = 'pgd';
    local_options.alpha = 1;
    local_options.tol_grad_ratio = 0.00001;
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);      
    

    if ~strcmp(options.alg, 'pgd') && ~strcmp(options.alg, 'direct_pgd')
        fprintf('Invalid algorithm: %s. Therfore, we use pgd (i.e., projected gradient descent).\n', options.alg);
        options.alg = 'pgd';
    else
        options.alg = options.alg;
    end

    if options.verbose > 0
        fprintf('# PGD (%s): started ...\n', options.alg);           
    end      
    
    % initialize factors
    init_options = options;
    [init_factors, ~] = generate_init_factors(V, rank, init_options);    
    W = init_factors.W;
    H = init_factors.H;      
        
    % initialize
    epoch = 0;    
    R_zero = zeros(m, n);
    grad_calc_count = 0; 
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);     

    tol_grad_ratio = options.tol_grad_ratio;
    if strcmp(options.alg, 'pgd')
        tol_grad_ratio = 0.00001; % tol = [0.001; 0.0001; 0.00001];
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;   
        init_grad = norm([gradW; gradH'],'fro');
        tolW = max(0.001,tol_grad_ratio)*init_grad; 
        tolH = tolW;
    elseif strcmp(options.alg, 'direct_pgd')
        tol_grad_ratio = 0.00001; % tol = [0.001; 0.0001; 0.00001];
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;           
        init_grad = norm([gradW; gradH'],'fro');    
        H = nlssubprob(V,W,H,0.001,1000);    
        obj = 0.5*(norm(V-W*H,'fro')^2);
        alpha = options.alpha;
    end    
    
    % store initial info
    clear infos;
    [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, [], epoch, grad_calc_count, 0);
    
    if options.verbose > 1
        fprintf('PGD (%s): Epoch = 0000, cost = %.16e, optgap = %.4e\n', options.alg, f_val, optgap); 
    end  
    
    % set start time
    start_time = tic();
    
    end_flag = 0;

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch) && ~end_flag        

        if strcmp(options.alg, 'pgd')
            
            % stopping condition
            projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
            if projnorm < tol_grad_ratio*init_grad
                end_flag = 1;
            end
  
            [W, gradW, iterW] = nlssubprob(V', H', W', tolW, 1000); 
            W = W'; 
            gradW = gradW';

            if iterW == 1
                tolW = 0.1 * tolW;
            end

            [H, gradH, iterH] = nlssubprob(V, W, H, tolH, 1000);
            if iterH == 1
                tolH = 0.1 * tolH; 
            end     

        elseif strcmp(options.alg, 'direct_pgd')

            gradW = W*(H*H') - V*H';
            gradH = (W'*W)*H - W'*V;

            projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);  
            if projnorm < tol_grad_ratio*init_grad
                fprintf('final grad norm %f\n', projnorm);
            else
                Wn = max(W - alpha*gradW,0);    
                Hn = max(H - alpha*gradH,0);    
                newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);

                if newobj-obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ sum(sum(gradH.*(Hn-H))))
                    % decrease stepsize    
                    while 1
                        alpha = alpha/10;
                        Wn = max(W - alpha*gradW,0);    
                        Hn = max(H - alpha*gradH,0);    
                        newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);

                        if newobj - obj <= 0.01*(sum(sum(gradW.*(Wn-W)))+ sum(sum(gradH.*(Hn-H))))
                            W = Wn; H = Hn;
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
                        Wn = max(W - alpha*gradW,0);    
                        Hn = max(H - alpha*gradH,0);    
                        newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);

                        %if (newobj - obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
                        %    sum(sum(gradH.*(Hn-H))))) | (Wn==Wp & Hn==Hp)
                        if (newobj - obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ sum(sum(gradH.*(Hn-H))))) ...
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

        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_infos(V, W, H, R_zero, options, infos, epoch, grad_calc_count, elapsed_time); 
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)
                fprintf('PGD (%s): Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if options.verbose > 0
        if optgap < options.tol_optgap
            fprintf('# PGD (%s): Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', options.alg, f_val, f_opt, options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('# PGD (%s): Max epoch reached (%g).\n', options.alg, options.max_epoch);
        end 
    end
    
    x.W = W;
    x.H = H;

end