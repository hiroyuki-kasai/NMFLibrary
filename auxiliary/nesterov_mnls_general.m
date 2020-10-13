function [A, iter, alpha] = nesterov_mnls_general(X, C, B, A_star, alpha, max_iter, func_type)
% Accelerated proximal gradient algorithm for nonnegative constraint
%   min_A f(A) := 1/2 | X - C*A*B' |^2_F,
%   s.t. A >= 0.
%
%   f(A)' = -C'*(X - C*A*B')*B = C'*C*A*B'*B - C'*X*B
%

    tol_1 = 1e-2;
    tol_2 = 1e-2;

    R = size(B,2);
    
    if ~isempty(B) && isempty(C)
        BTB = B' * B;
        W = - X * B;
    elseif isempty(B) && ~isempty(C)
        CTC = C' * C;
        W = - C'* X;
    elseif ~isempty(B) && ~isempty(C)
        BTB = B' * B;
        CTC = C' * C;
        W = - C'* X * B;
    else
        
    end    

    
    if strcmp(func_type, 'smooth') || strcmp(func_type, 'stochastic') ||  strcmp(func_type, 'strong_alpha_beta') 
        
        if ~isempty(B) && isempty(C)
            s = svd(BTB);
        elseif isempty(B) && ~isempty(C)
            s = svd(CTC);
        elseif ~isempty(B) && ~isempty(C)
            % to do
        else
        end
        
        L = max(s);
        mu = min(s);        
    
    else
        if ~isempty(B) && isempty(C)
            L = norm(BTB);
        elseif isempty(B) && ~isempty(C)
            L = norm(CTC);
        elseif ~isempty(B) && ~isempty(C)
            L = norm(CTC)*norm(BTB);
        else
        end
          
    end
    

    if strcmp(func_type, 'smooth') || strcmp(func_type, 'stochastic')
        q = mu/L; 
    elseif strcmp(func_type, 'strong_alpha_beta') 
        lambda = g_lambda(L, mu);
        BTB = BTB + lambda * eye(R);  
        W = W - lambda * A_star;
        L = L + lambda;
        q = (mu + lambda)/L;   
    else
    end
    
    
    Y = A_star;
    A_prev = Y;
    
    %alpha = 1;
    iter = 0;
    while iter < max_iter
        
        % calculate gradient
        if ~isempty(B) && isempty(C)
            grad = Y * BTB + W;
        elseif isempty(B) && ~isempty(C)
            grad = CTC * Y + W;
        elseif ~isempty(B) && ~isempty(C)
            grad = CTC * Y * BTB + W;
        else
        end        

        % terminate the loop
        cond_1 = max(max( abs(grad .* Y)));
        cond_2 = min((min (grad)));
        if cond_1 < tol_1 && cond_2 > - tol_2 && iter > 0
            break
            %iter = iter + 1;
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
    
    %A = Y;
    
end

