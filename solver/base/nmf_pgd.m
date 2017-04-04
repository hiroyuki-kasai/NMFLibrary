function [x, infos] = nmf_pgd(V, rank, options)
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
%       options 
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
% Modified by H.Kasai on Apr. 04, 2017

    m = size(V, 1);
    n = size(V, 2); 
    
    if ~isfield(options, 'alg')
        alg = 'pgd';
    else
        if ~strcmp(options.alg, 'pgd') && ~strcmp(options.alg, 'direct_pgd')
            fprintf('Invalid algorithm: %s. Therfore, we use pgd (i.e., projected gradient descent).\n', options.alg);
            alg = 'pgd';
        else
            alg = options.alg;
        end
    end     

    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end   
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end       

    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end

    if ~isfield(options, 'x_init')
        W = rand(m, rank);
        H = rand(rank, n);
    else
        W = options.x_init.W;
        H = options.x_init.H;
    end 
    
    % initialize
    epoch = 0;    
    R = zeros(m, n);
    grad_calc_count = 0; 
    
    % store initial info
    clear infos;
    infos.epoch = 0;
    f_val = nmf_cost(V, W, H, R);
    infos.cost = f_val;
    optgap = f_val - f_opt;
    infos.optgap = optgap;   
    infos.time = 0;
    infos.grad_calc_count = grad_calc_count;
    if verbose > 0
        fprintf('PGD (%s): Epoch = 000, cost = %.16e, optgap = %.4e\n', alg, f_val, optgap); 
    end  
    
    % select disp_freq 
    if verbose > 0
        disp_freq = floor(max_epoch/100);
        if disp_freq < 1 || max_epoch < 200
            disp_freq = 1;
        end
    end    

    if strcmp(alg, 'pgd')
        tol = 0.00001; % tol = [0.001; 0.0001; 0.00001];
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;   
        initgrad = norm([gradW; gradH'],'fro');
        tolW = max(0.001,tol)*initgrad; 
        tolH = tolW;
    elseif strcmp(alg, 'direct_pgd')
        %tol = 0.00001; % tol = [0.001; 0.0001; 0.00001];
        alpha = 1;
        gradW = W*(H*H') - V*H'; 
        gradH = (W'*W)*H - W'*V;           
        %initgrad = norm([gradW; gradH'],'fro');    
        %fprintf('init grad norm %f\n', initgrad);
        H = nlssubprob(V,W,H,0.001,1000);    
        obj = 0.5*(norm(V-W*H,'fro')^2);            
    end
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)           

        if strcmp(alg, 'pgd')
          [W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000); 
          W = W'; 
          gradW = gradW';
          
          if iterW==1,
            tolW = 0.1 * tolW;
          end

          [H,gradH,iterH] = nlssubprob(V,W,H,tolH,1000);
          if iterH==1,
            tolH = 0.1 * tolH; 
          end     

        elseif strcmp(alg, 'direct_pgd')

          gradW = W*(H*H') - V*H';
          gradH = (W'*W)*H - W'*V;

          projnorm = norm([gradW(gradW<0 | W>0); 
          gradH(gradH<0 | H>0)]);  

          Wn = max(W - alpha*gradW,0);    
          Hn = max(H - alpha*gradH,0);    
          newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
          if newobj-obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
                                sum(sum(gradH.*(Hn-H)))),
            % decrease alpha    
            while 1,
              alpha = alpha/10;
              Wn = max(W - alpha*gradW,0);    
              Hn = max(H - alpha*gradH,0);    
              newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
              if newobj - obj <= 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
                                       sum(sum(gradH.*(Hn-H)))),
                W = Wn; H = Hn;
                obj = newobj;
                break;
              end
            end
          else 
            % increase alpha
            while 1,
              Wp = Wn; Hp = Hn; objp = newobj;
              alpha = alpha*10;
              Wn = max(W - alpha*gradW,0);    
              Hn = max(H - alpha*gradH,0);    
              newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
              if (newobj - obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
                                      sum(sum(gradH.*(Hn-H))))) | (Wn==Wp & Hn==Hp),
                W = Wp; 
                H = Hp;
                obj = objp; 
                alpha = alpha/10;
                break;
              end
            end
          end            
          
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % calculate cost and optgap 
        f_val = nmf_cost(V, W, H, R);
        optgap = f_val - f_opt;    
        
        % update epoch
        epoch = epoch + 1;         
        
        % store info
        infos.epoch = [infos.epoch epoch];
        infos.cost = [infos.cost f_val];
        infos.optgap = [infos.optgap optgap];     
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % display infos
        if verbose > 0
            if ~mod(epoch, disp_freq)
                fprintf('PGD (%s): Epoch = %03d, cost = %.16e, optgap = %.4e\n', alg, epoch, f_val, optgap);
            end
        end        
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', f_val, f_opt, tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', max_epoch);
    end 
    
    x.W = W;
    x.H = H;

end