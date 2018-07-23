function f = nmf_cost(V, W, H, R, varargin)
% Calculate the cost function of NMF.
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai on Feb. 21, 2017
% Modified by H.Kasai on Jul. 23, 2018

    if isempty(varargin) || strcmp(varargin{1}, 'EUC')
        f = norm(V - W * H - R,'fro')^2 / 2 ;
        %f = sum(sum((V-W * H - R).^2)) / 2; 
    elseif strcmp(varargin{1}, 'KL')
        Vhat = W * H - R;
        
        temp = V.*log(V./Vhat);
        temp(temp ~= temp) = 0; % NaN ~= NaN
        f = sum(sum(temp - V + Vhat));    
    else
        % use 'EUC'
        f = norm(V - W * H - R,'fro')^2 / 2 ;
    end
    
end
