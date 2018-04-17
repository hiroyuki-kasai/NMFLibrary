function [p,dist]=purity_f(indCluster,classes)
% compute purity of clustering according to class information
% indCluster: column vector, the cluster index of each data point
% classes: column vector, the class label of each data point
% p: scalar, purity
% dist: the distribution matrix of clustering
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
% May 01, 2011
%%%%

numSample=numel(classes);
uniClass=unique(classes);
numClass=numel(uniClass);
uniCluster=unique(indCluster);
numCluster=numel(uniCluster);
dist=zeros(numCluster,numClass);

for i=1:numCluster
    curCla=(classes==uniClass(i));
    for j=1:numClass
        curClu=(indCluster==(uniCluster(j)));
        dist(i,j)=sum(curClu&curCla);      
    end
end

p=sum(max(dist,[],2))/numSample;

end