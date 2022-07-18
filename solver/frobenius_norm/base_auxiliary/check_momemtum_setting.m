function [options, beta, betamax] = check_momemtum_setting(options)

    if options.momentum_w > 0 || options.momentum_h > 0
        options.sub_mode = 'momentum';        

        if options.eta < options.gammabeta || options.gammabeta < options.gammabetabar
            error('You should choose eta > gamma > gammabar.');
        end
        if options.beta0 > 1 || options.beta0 < 0
            error('beta0 must be in the interval [0,1].');
        end 

        if options.momentum_h > 3
            error('momentum_h (%d) should be <= 3\n', options.momentum_h);
        end

        if options.momentum_w ~= 0 && options.momentum_w ~= 2
            error('momentum_w (%d) should be 0 or 2.\n', options.momentum_w);
        end
    
        beta = zeros(1, options.max_epoch);    
        beta(1) = options.beta0; 
        betamax = 1; 
        options.warm_restart = true;        
    else
        beta = 0;  
        betamax = 0;    
        options.warm_restart = false;  % we do not need warm_restart when non-momentum
    end 
end

