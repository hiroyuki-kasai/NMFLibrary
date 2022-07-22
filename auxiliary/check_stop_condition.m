function [stop_flag, reason, max_reached_flag, rev_info] = check_stop_condition(epoch, info, options, stop_options)
% Function to check stop condition
%
% Inputs:
%       epoch               epoch number
%       info                statistics of solver
%       options             options
%       stop_options        additional options for stopping condition
% Output:
%       stop_flag           0: not stopping, >0: stop due to some reason
%       reason              description of the reason of stopping
%       max_reached_flag    1: reached a given max_epoch
%       rev_info            modified statistics of solver
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 14, 2022
%


    if nargin < 4
        stop_options = [];
    end
    
    stop_flag = 0;
    reason = [];
    max_reached_flag = false;
    rev_info = info;


    if ~options.not_store_infos    
        if info.optgap(end) < options.tol_optgap
            stop_flag = 1;
            reason = sprintf('Optimality gap tolerance reached: f_val = %.4e < f_opt = %.4e (%.4e)\n', info.cost(end), options.f_opt, options.tol_optgap);
            return;
        end
    end
    
    if epoch >= options.max_epoch    
        stop_flag = 2;
        max_reached_flag = true;
        reason = sprintf('Max epoch reached (%g).\n', options.max_epoch);
        return;        
    end

    if isfield(options, 'special_stop_condition') && ~isempty(stop_options)
        [stop_flag, reason, rev_info] = options.special_stop_condition(epoch, info, options, stop_options);
        if stop_flag
            stop_flag = 3;
            return;              
        end
    end

end