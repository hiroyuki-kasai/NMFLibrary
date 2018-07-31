function [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time, metric, metric_param)
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
%       metric          'EUC', 'KL', 'ALPHA-D', 'BETA-D'
%       metric_param    alpha, beta
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Oct. 27, 2017
% Created by H.Kasai on Jul. 26, 2018

    f_val = Inf;
    optgap = Inf;

    if nargin < 10
        metric = 'EUC';
        metric_param = [];
    elseif nargin < 11
        metric = 'KL';
        metric_param = [];        
    end


    if ~epoch
        
        infos.iter = epoch;
        infos.epoch = epoch;        
        infos.time = 0;    
        infos.grad_calc_count = grad_calc_count;
        if ~isempty(V) || ~isempty(W) || ~isempty(H) || ~isempty(R)
            f_val = nmf_cost(V, W, H, R, metric, metric_param);
            optgap = f_val - options.f_opt;
            infos.optgap = optgap;
            infos.cost = f_val;
        end
    
        if options.store_sol
            if ~isempty(W) infos.W = [infos.W W];   end
            if ~isempty(H) infos.H = [infos.H H];   end
            if ~isempty(R) infos.R = [infos.R R];   end   
        end
        
    else
        
        infos.iter = [infos.iter epoch];
        infos.epoch = [infos.epoch epoch];        
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        if ~isempty(V) || ~isempty(W) || ~isempty(H) || ~isempty(R)        
            f_val = nmf_cost(V, W, H, R, metric, metric_param);
            optgap = f_val - options.f_opt;  
            infos.optgap = [infos.optgap optgap];
            infos.cost = [infos.cost f_val];
        end

         
        if options.store_sol
            if ~isempty(W) infos.W = [infos.W W];   end
            if ~isempty(H) infos.H = [infos.H H];   end
            if ~isempty(R) infos.R = [infos.R R];   end           
        end  
        
    end

end

