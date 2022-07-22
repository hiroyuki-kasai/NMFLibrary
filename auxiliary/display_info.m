function [] = display_info(method_name, epoch, nmf_info, options)
% Function to display statistics of solver
%
% Inputs:
%       method_name         name of solver (text)
%       epoch               epoch number
%       nmf_info            statistics of solver
%       options             options
% Output:
%
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 14, 2022
%


    if options.disp_freq == 0
        % select disp_freq 
        options.disp_freq = set_disp_frequency(options);     
    end


    if ~options.not_store_infos  

        f_val = nmf_info.cost(end);
        optgap = nmf_info.optgap(end);
        process_time = nmf_info.time(end) - nmf_info.time(end-1);
    
        % display infos
        if options.verbose > 1
            if ~mod(epoch, options.disp_freq)
                fprintf('%s: Epoch = %04d, cost = %.16e, optgap = %.4e, time = %e\n', method_name, epoch, f_val, optgap, process_time);
            end
        elseif options.verbose == 1
    
            if strcmp(method_name, 'SPA') || strcmp(method_name, 'SNPA')
                textwaitbar(epoch, options.max_epoch-1, '  progress');
            else
                textwaitbar(epoch, options.max_epoch, '  progress');            
            end
            
        end
    else

        % display infos
        if options.verbose > 1
            if ~mod(epoch, options.disp_freq)
                fprintf('%s: Epoch = %04d\n', method_name, epoch);
            end
        elseif options.verbose == 1
    
            if strcmp(method_name, 'SPA') || strcmp(method_name, 'SNPA')
                textwaitbar(epoch, options.max_epoch-1, '  progress');
            else
                textwaitbar(epoch, options.max_epoch, '  progress');            
            end
            
        end

    end

end