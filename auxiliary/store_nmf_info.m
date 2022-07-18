function [infos, f_val, optgap] = store_nmf_info(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time)
% Function to store statistic information
%
% Inputs:
%       V               (m x n) non-negative matrix to factorize
%       W               (m x r) non-negative factor matrix (solution)
%       H               (r x n) non-negative factor matrix (solution)
%       R               (m x n) non-negative outlier matrix
%       options         options
%       infos           struct to store statistic information
%       epoch           number of outer iteration
%       grad_calc_count number of calclations of gradients
%       elapsed_time    elapsed time from the begining
%       metric          'euc', 'kl-div', 'alpha-div', 'beta-div'
%       metric_param    alpha, beta
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Oct. 27, 2017
%
%   Jul. 26, 2018 (Hiroyuki Kasai): Fixed algorithm. 
%
%   Jun. 25, 2019 (Hiroyuki Kasai): Added clustering accuracy measurement 
%                                   and symmetry evaluation.
%
%   Jun. 30, 2022 (Hiroyuki Kasai): Done refactoring.
%

    f_val = Inf;
    optgap = Inf;  

    if options.not_store_infos

        % Do nothing.

    else

        if ~epoch
            
            infos.iter = epoch;
            infos.epoch = epoch;        
            infos.time = 0;    
            infos.grad_calc_count = grad_calc_count;
            if ~isempty(V) || ~isempty(W) || ~isempty(H) || ~isempty(R)
                if ~isfield(options, 'special_nmf_cost')
                    f_val = nmf_cost(V, W, H, R, options);
                else
                    f_val = options.special_nmf_cost(V, W, H, R, options);
                end
                optgap = f_val - options.f_opt;
                infos.optgap = optgap;
                infos.cost = f_val;
                infos.cost_best = f_val;
            end
        
            if options.store_sol
                if ~isempty(W) 
                    infos.W = W;   
                end
                if ~isempty(H) 
                    infos.H = H;   
                end
                if ~isempty(R) 
                    infos.R = R;   
                end   
            end
            
            if options.calc_symmetry
                infos.symmetry = norm(W-H','fro')^2;
            end
            
            if options.calc_clustering_acc && ~isempty(options.clustering_gnd) && options.clustering_classnum > 0
                infos.clustering_acc = eval_clustering_accuracy(H, options.clustering_gnd, options.clustering_classnum, options.clustering_eval_num);        
            end
            
        else
            
            infos.iter = [infos.iter epoch];
            infos.epoch = [infos.epoch epoch];        
            infos.time = [infos.time elapsed_time];
            infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
            if ~isempty(V) || ~isempty(W) || ~isempty(H) || ~isempty(R)        
                if ~isfield(options, 'special_nmf_cost')
                    f_val = nmf_cost(V, W, H, R, options);
                else
                    f_val = options.special_nmf_cost(V, W, H, R, options);
                end
                optgap = f_val - options.f_opt;  
                infos.optgap = [infos.optgap optgap];
                infos.cost = [infos.cost f_val];
                if infos.cost_best(end) > f_val
                    infos.cost_best = [infos.cost_best f_val];    
                else
                    infos.cost_best = [infos.cost_best infos.cost_best(end)];
                end
            end
    
            if options.store_sol
                if ~isempty(W) 
                    infos.W = [infos.W W];   
                end
                if ~isempty(H) 
                    infos.H = [infos.H H];   
                end
                if ~isempty(R) 
                    infos.R = [infos.R R];   
                end           
            end  
            
            if options.calc_symmetry
                infos.symmetry = [infos.symmetry norm(W - H', 'fro')^2];
            end        
            
            if options.calc_clustering_acc && ~isempty(options.clustering_gnd) && options.clustering_classnum > 0   
                infos.clustering_acc = [infos.clustering_acc eval_clustering_accuracy(H, options.clustering_gnd, options.clustering_classnum, options.clustering_eval_num)];
            end  

        end

    end

    infos.final_W = W;
    infos.final_H = H;
    infos.final_R = R;    

end

