function [W, H, W_prev, H_prev, rel_error, beta, betamax, options] = warm_restart(V, W, H, rank, W_prev, H_prev, W_tmp1, H_tmp1, H_tmp2, rel_error, beta, betamax, epoch, options)

    nV = norm(V, 'fro');
    
    % check relative error 
    rel_error(epoch+2) = max(0,nV^2 - 2*sum(sum(V*H_tmp2' .* W_tmp1)) + sum(sum(H_tmp2*H_tmp2' .* (W_tmp1'*W_tmp1))));
    rel_error(epoch+2) = sqrt(rel_error(epoch+2))/nV;         
    if options.verbose > 2        
        fprintf('\trel_error(i+1) = %.5e, e(i) = %.5e\n', rel_error(epoch+2), rel_error(epoch+1));        
    end
    
    f_val_prev = nmf_cost(V, W_prev, H_prev, []);
    f_val_curr = nmf_cost(V, W, H, []);    

    %if rel_error(epoch+2) >= rel_error(epoch+1) + 10*eps    
    if f_val_prev <= f_val_curr - 10*eps

        options.delta = options.delta/10;
        options.inner_max_epoch = ceil(1.5 * options.inner_max_epoch);  

        W = W_prev;             
        H = H_prev;

        if options.verbose > 1
            fprintf('\twarm_restarting: relative error increased +%.16e\n', rel_error(epoch+2) - rel_error(epoch+1));
        end

        if options.scaling            
            normW = sqrt((sum(W.^2))) + 1e-16;
            normH = sqrt((sum(H'.^2))) + 1e-16;
            for k = 1 : rank
                W(:,k) = W(:,k) / sqrt(normW(k)) * sqrt(normH(k));
                H(k,:) = H(k,:) / sqrt(normH(k)) * sqrt(normW(k));
            end 
        end

        W_prev = W;
        H_prev = H;

        if strcmp(options.sub_mode, 'momentum')
            if epoch == 2
                betamax  = beta(epoch+1); 
            else
                betamax  = beta(epoch); 
            end
            beta(epoch+2) = beta(epoch+1)/options.eta; 
        end

    else

        if strcmp(options.sub_mode, 'momentum')            
            W_prev = W_tmp1;            
            H_prev = H_tmp1;            
            beta(epoch+2) = min(betamax, beta(epoch+1)*options.gammabeta); 
            betamax = min(1, betamax*options.gammabetabar); 
        else
            W_prev = W;            
            H_prev = H;            
        end        
        
    end
end
