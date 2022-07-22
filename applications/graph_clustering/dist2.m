function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%   D(i,j) = norm(X(i,:)-C(j,:)).^2;
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

%	Dec 6, 2011: Modified by Da Kuang, with performance improvement.

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

tempx = full(sum(x.^2, 2));
tempc = full(sum(c.^2, 2)');

n2 = tempx(:, ones(1,ncentres)) + ...
  		tempc(ones(1,ndata), :) - ...
  		2.*(x*(c'));

