function [newL, newR] = colsum_L_one(L, R, varargin)
% normalizes columns in L so that the product LR stays the same

    if isempty(varargin)
        [helpR, helpL] = rowsum_R_one(R',L');
    else
        [helpR, helpL] = rowsum_R_one(R',L',varargin{1});
    end
    
    newR = helpR';
    newL = helpL';

end

