function [newL, newR] = rowsum_R_one(L, R, varargin)
% normalizes rows in R so that the product LR stays the same
% handle_zeros option: leaves rows that sum up to 0 as they were

    coeffL = sum(R,2);
    
    if not(isempty(varargin)) && varargin{1}==1
        coeffL(coeffL==0) = 1;
    end
    
    coeffR = 1 ./ coeffL;
    left = diag(coeffL);
    right = diag(coeffR);
    newL = L*left;
    newR = right*R;
    
end

