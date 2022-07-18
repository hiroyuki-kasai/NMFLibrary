function f = nmf_cost(V, W, H, R, options)
% Calculate the cost function of NMF.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Feb. 21, 2017
%
% Change log: 
%
%   Jul. 23, 2018 (Hiroyuki Kasai): Fixed algorithm.
%
%   Jun. 30, 2022 (Hiroyuki Kasai): Done refactoring.
%


    if nargin < 4 || isempty(R)
        R = zeros(size(V));
    end
    
    if nargin < 5
        options.metric_type = 'euc';
    end    

    Vhat = W * H + R;

    if strcmp(options.metric_type, 'euc')
        
        f = norm(V - Vhat, 'fro')^2 / 2 ;
        %f = sum(sum((V - Vhat).^2)) / 2; 
        
    elseif strcmp(options.metric_type, 'kl-div')
        
        %Vhat = W * H + R;
        Vhat = Vhat + (Vhat<eps) .* eps;
        
        temp = V.*log(V./Vhat);
        temp(temp ~= temp) = 0; % NaN ~= NaN
        f = sum(sum(temp - V + Vhat)); 
        
    elseif strcmp(options.metric_type, 'alpha-div')
        
        alpha = options.d_alpha;
        
        f = sum(V(:).^alpha .* Vhat(:).^(1-alpha) - alpha*V(:) + ...
                  (alpha-1)*Vhat(:)) / (alpha*(alpha-1));
    
    elseif strcmp(options.metric_type, 'beta-div')
        
        beta = options.d_beta;
        
        switch beta
            case 0  % equivalent to IS (Itakura-Sato) divergence
                f = sum(V(:)./Vhat(:) - log(V(:)./Vhat(:)) - 1);     
            case 1  % equivalent to KL divergence
                f = sum(V(:).*log(V(:)./Vhat(:)) - V(:) + Vhat(:));
            case 2  % equivalent to Frobenius norm
                f = norm(V - Vhat, 'fro')^2 / 2 ;   
            otherwise
                f = sum(V(:).^beta + (beta-1)*Vhat(:).^beta - beta*V(:).*Vhat(:).^(beta-1)) / ...
                          (beta*(beta-1));
        end

    elseif strcmp(options.metric_type, 'is-div')  
        f = sum(V(:)./Vhat(:) - log(V(:)./Vhat(:)) - 1);
    else
        % use 'euc'
        f = norm(V - Vhat, 'fro')^2 / 2 ;
    end
    
end
