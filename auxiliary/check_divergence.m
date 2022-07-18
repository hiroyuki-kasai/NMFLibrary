function [options] = check_divergence(options)
% Function to chek divergence
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on June 29, 2022
%


    metric_type = options.metric_type;
    
    if strcmp(options.metric_type, 'euc')    
        metric_param = 2;
    elseif strcmp(options.metric_type, 'kl-div')    
        metric_param = 1;
    elseif strcmp(options.metric_type, 'is-div')    
        metric_param = 0;
    elseif strcmp(options.metric_type, 'alpha-div')
        metric_param = options.d_alpha;
    elseif strcmp(options.metric_type, 'beta-div')
        metric_param = options.d_beta;
    else
        metric_param = 2;
        metric_type = 'euc';         
    end

    options.metric_type = metric_type;
    options.metric_param = metric_param;    
end