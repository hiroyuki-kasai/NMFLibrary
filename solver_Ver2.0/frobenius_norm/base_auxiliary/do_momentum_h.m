function [H, H_tmp1, H_tmp2] = do_momentum_h(H, H_prev, beta, epoch, options) 

    H_tmp1 = H;
                
    if strcmp(options.sub_mode, 'momentum')        
        % perform extrapolation
        if options.momentum_h >= 2 
            H = H + beta(epoch+1) * (H - H_prev) ;
        end

        % project
        if options.momentum_h == 3
            H = max(0, H); 
        end            
    end
    H_tmp2 = H; 

    if options.verbose > 2        
        fprintf('\tnorm_Hn = %.10e, norm_Hy = %.10e, norm_H = %.10e, norm_Wy = %.10e\n', norm(H_tmp1), norm(H), norm(H_prev), norm(W));        
    end      
end
