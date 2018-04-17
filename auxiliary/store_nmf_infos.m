function [infos, f_val, optgap] = store_nmf_infos(V, W, H, R, options, infos, epoch, grad_calc_count, elapsed_time)
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
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Oct. 27, 2017


    if ~epoch
        
        infos.iter = epoch;
        infos.epoch = epoch;        
        infos.time = 0;    
        infos.grad_calc_count = grad_calc_count;
        f_val = nmf_cost(V, W, H, R);
        optgap = f_val - options.f_opt;
        infos.optgap = optgap;
        infos.cost = f_val;
    
        if options.store_sol
            infos.W = W;   
            infos.H = H;
            infos.R = R;
        end
        
    else
        
        infos.iter = [infos.iter epoch];
        infos.epoch = [infos.epoch epoch];        
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        f_val = nmf_cost(V, W, H, R);
        optgap = f_val - options.f_opt;  
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];

         
        if options.store_sol
            infos.W = [infos.W W];   
            infos.H = [infos.H H];
            infos.R = [infos.R R];           
        end  
        
    end

end

