% Compute average Hoyer sparsity of columns of W

function spx = sp_col(X,w) 

[m,r] = size(X); 
for i = 1 : r
    x{i} = X(:,i); 
    if nargin == 1
        w{i} = ones(m,1);
    end
end
spx = sp(x,w); 