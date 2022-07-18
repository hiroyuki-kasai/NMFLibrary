function [W, H, W_tmp1] = do_momentum_w(W, W_prev, H, H_prev, H_tmp1, beta, epoch, options) 

    W_tmp1 = W;
        
    if strcmp(options.sub_mode, 'momentum')        
        if options.momentum_w == 2 
            W = W + beta(epoch+1) * (W - W_prev); 
        end   

        if options.momentum_h == 1
            H =  H_tmp1 + beta(epoch+1)*(H_tmp1 - H_prev) ;
        end
    end 

    if options.verbose > 2
        fprintf('\tnorm_Wn = %.10e, norm_Wy = %.10e, norm_Hy = %.10e, beta = %.2f\n', norm(W_tmp1), norm(W), norm(H), beta(epoch+1));        
    end  
            
end