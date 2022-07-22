function k=kernelSigmoid(x,y,param)
% sigmoid kernel sigmoid(x,y)=tanh(alpha*(x'*y) + beta)
% Usage:
% k=kernelSigmoid(x,y)
% k=kernelSigmoid(x,y,param)
% x,y: column vectors, or matrices.
% param: scalar, [alpha;beta].
% k, scalar or matrix, the kernel values
% Yifeng Li, May 26, 2011.
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

k=x'*y;
if nargin<3
   alpha=1;
   beta=0;
else
    alpha=param(1);
    beta=param(2);
end
k=tanh(alpha*k + beta);
end