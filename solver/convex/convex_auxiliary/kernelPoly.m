function k=kernelPoly(x,y,param,varargin)
% Polynomial kernel k=(Gamma.*(x'*y)+ Coefficient).^Degree;
% Usage:
% k=kernelPoly(x,y)
% k=kernelPoly(x,y,param)
% x,y: column vectors, or matrices.
% param: [Gamma;Coefficient;Degree]
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
    Gamma=1;
    Coefficient=0;
    Degree=2;
else
    Gamma=param(1);
    Coefficient=param(2);
    Degree=param(3);
end
k=(Gamma.*(x'*y)+ Coefficient).^Degree;
end