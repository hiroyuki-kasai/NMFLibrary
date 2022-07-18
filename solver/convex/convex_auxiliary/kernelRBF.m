function k=kernelRBF(x,y,param)
% rbf kernel rbf(x,y)=exp((-(1/sigma^2).*(|x-y|.^2));
% Usage:
% k=kernelRBF(x,y)
% k=kernelRBF(x,y,param)
% x,y: column vectors, or matrices.
% param: scalar, [sigma].
% k, scalar or matrix, the kernel values
%%%%
% Copyright (C) <2012>  <Yifeng Li>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 26, 2011
%%%%


if nargin<3
    sigma=1;
else
    sigma=param(1,1);
    if sigma==0
        error('sigma must not be zero!');
    end
end
k=exp((-(1/sigma^2)).*(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1)));
end