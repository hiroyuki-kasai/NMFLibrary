function disp_freq = set_disp_frequency(options)
% Function to set the frequency of statistic display
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai
%


    % select disp_freq 
    if options.verbose > 0
        disp_freq = floor(options.max_epoch/100);
        if disp_freq < 1 || options.max_epoch < 200
            disp_freq = 1;
        end
    else
        disp_freq = 100000;
    end  

end

