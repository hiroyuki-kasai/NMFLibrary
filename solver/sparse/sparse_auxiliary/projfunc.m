function [v,usediters] = projfunc( s, k1, k2, nn )

% Solves the following problem:
% Given a vector s, find the vector v having sum(abs(v))=k1 
% and sum(v.^2)=k2 which is closest to s in the euclidian sense.
% If the binary flag nn is set, the vector v is additionally
% restricted to being non-negative (v>=0).
%    
% Written 2.7.2004 by Patrik O. Hoyer
%
    
% Problem dimension
N = length(s);

% If non-negativity flag not set, record signs and take abs
if ~nn,
    isneg = s<0;
    s = abs(s);
end

% Start by projecting the point to the sum constraint hyperplane
v = s + (k1-sum(s))/N; 

% Initialize zerocoeff (initially, no elements are assumed zero)
zerocoeff = [];

j = 0;
while 1,

    % This does the proposed projection operator
    midpoint = ones(N,1)*k1/(N-length(zerocoeff)); 
    midpoint(zerocoeff) = 0;
    w = v-midpoint;
    a = sum(w.^2); 
    b = 2*w'*v;
    c = sum(v.^2)-k2;
    alphap = (-b+real(sqrt(b^2-4*a*c)))/(2*a); 
    v = alphap*w + v;
    
    if all(v>=0),
	% We've found our solution
	usediters = j+1;
	break;
    end
        
    j = j+1;
        
    % Set negs to zero, subtract appropriate amount from rest
    zerocoeff = find(v<=0);
    v(zerocoeff) = 0;
    tempsum = sum(v);
    v = v + (k1-tempsum)/(N-length(zerocoeff));
    v(zerocoeff) = 0;
            
end

% If non-negativity flag not set, return signs to solution
if ~nn,
    v = (-2*isneg + 1).*v;
end

% Check for problems
if max(max(abs(imag(v))))>1e-10,
    error('Somehow got imaginary values!');
end
