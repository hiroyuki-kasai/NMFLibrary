function k=kernelLinear(x,y,param)
% linear kernel
% Usage:
% k=kernelLinear(x,y)
% x,y: column vectors, or matrices.
% param: []
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

if size(x,3)>1
    x=matrizicing(x,3);
    y=matrizicing(y,3);
    x=x';
    y=y';   
end
k=x'*y;
end