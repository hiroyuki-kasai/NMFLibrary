function [A, iter, alpha] = nesterov_mnls(X, B, A_star, alpha, max_iter, func_type)

    tol_1 = 1e-2;
    tol_2 = 1e-2;

    R = size(B,2);
    Z = B' * B;
    
    s = svd(Z);
    L = max(s);
    %L_new = norm(BtB, 'fro');  
    
    mu = min(s);
    
    if strcmp(func_type, 'smooth') || strcmp(func_type, 'stochastic')
        W = - X * B;
        q = mu/L; 
    elseif strcmp(func_type, 'strong_alpha_beta') 
        lambda = g_lambda(L, mu);
        Z = Z + lambda * eye(R);  
        W = - X * B - lambda * A_star;
        L = L + lambda;
        q = (mu + lambda)/L;   
    else
        W = - X * B;        
    end
    
    
    Y = A_star;
    A_prev = Y;
    
    %alpha = 1;
    iter = 0;
    while iter < max_iter
        grad = W + Y * Z;

        % terminate the loop
        cond_1 = max(max( abs(grad .* Y)));
        cond_2 = min((min (grad)));
        if cond_1 < tol_1 && cond_2 > - tol_2 && iter > 0
            break        
        else
            A = Y - 1/L * grad;
            A = max(A, 0);            
            if strcmp(func_type, 'smooth') || strcmp(func_type, 'strong_alpha_beta')
                new_alpha = update_alpha(alpha, q);
                beta = alpha * (1 - alpha) / (alpha^2 + new_alpha);
                alpha = new_alpha;                
            elseif strcmp(func_type, 'stochastic')  
                beta = (1-sqrt(mu * 1/L))/(1+sqrt(mu * 1/L));
            elseif strcmp(func_type, 'basic')
                new_alpha = (1+sqrt(4 * alpha^2 + 1))/2;
                beta = (alpha-1)/new_alpha;
                alpha = new_alpha;                 
            else
            end
            Y = A + beta * (A - A_prev);

            A_prev = A;
            iter = iter + 1;
        end
    end
    
    A = Y;
    
end

