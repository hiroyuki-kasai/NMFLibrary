function [] = display_stop_reason(epoch, infos, options, disp_name, reason, max_reached_flag)
% Function to display stop reason
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on July 14, 2022
%

    
    if options.verbose > 0

        if options.verbose == 1 && ~max_reached_flag
            fprintf(' (incomplete)\n');
        end

        fprintf('# %s: %s', disp_name, reason);
    end

end