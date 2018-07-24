function W = normalize_W(W, type)
% function W = normalize_W(W, type)
%
% Normalize columns of W using:
%  1 - use 1-norm [default]
%  2 - use 2-norm
%  k - multiply the 1-norm by k
%
% This should work both for matrices and tensors (only for Convolutive NMF)
%
% 2010-01-14 Graham Grindlay (grindlay@ee.columbia.edu)

% Copyright (C) 2008-2028 Graham Grindlay (grindlay@ee.columbia.edu)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

if nargin < 2
    type = 1;
end

switch type
    case 1
        for j = 1:size(W,3)
            for i = 1:size(W,2)
                W(:,i,j) = W(:,i,j) ./ norm(W(:,i,j),1);
            end
        end
        
    case 2
        for j = 1:size(W,3)
            for i = 1:size(W,2)
                W(:,i,j) = W(:,i,j) ./ norm(W(:,i,j),2);
            end
        end
        
    case 0
        
    otherwise 
        for j = 1:size(W,3)
            for i = 1:size(W,2)
                W(:,i,j) = type*W(:,i,j) ./ norm(W(:,i,j),1);
            end
        end
end
        