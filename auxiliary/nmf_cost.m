function f = nmf_cost(V, W, H, R, varargin)
% Calculate the cost function of NMF.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Feb. 21, 2017
% Modified by H.Kasai on Jul. 23, 2018

    Vhat = W * H + R;

    if isempty(varargin) || strcmp(varargin{1}, 'EUC')
        
        f = norm(V - Vhat,'fro')^2 / 2 ;
        %f = sum(sum((V - Vhat).^2)) / 2; 
        
    elseif strcmp(varargin{1}, 'KL')
        
        Vhat = W * H + R;
        Vhat = Vhat + (Vhat<eps) .* eps;
        
        temp = V.*log(V./Vhat);
        temp(temp ~= temp) = 0; % NaN ~= NaN
        f = sum(sum(temp - V + Vhat)); 
        
    elseif strcmp(varargin{1}, 'ALPHA-D')
        
        alpha = varargin{2};
        
        f = sum(V(:).^alpha .* Vhat(:).^(1-alpha) - alpha*V(:) + ...
                  (alpha-1)*Vhat(:)) / (alpha*(alpha-1));
    
    elseif strcmp(varargin{1}, 'BETA-D')
        
        beta = varargin{2};
        
        switch beta
            case 0
                f = sum(V(:)./Vhat(:) - log(V(:)./Vhat(:)) - 1);     
            case 1
                f = sum(V(:).*log(V(:)./Vhat(:)) - V(:) + Vhat(:));
            case 2
                f = sum(sum((V-W*H).^2));
            otherwise
                f = sum(V(:).^beta + (beta-1)*Vhat(:).^beta - beta*V(:).*Vhat(:).^(beta-1)) / ...
                          (beta*(beta-1));
        end
    
    else
        % use 'EUC'
        f = norm(V - Vhat,'fro')^2 / 2 ;
    end
    
end
