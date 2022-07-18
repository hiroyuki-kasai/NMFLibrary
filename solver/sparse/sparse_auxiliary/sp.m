% Compute Hoyer sparsity of x 

function spx = sp(x,w) 

r = length(x); 

spx = 0; 
for i = 1 : r
    if x{i} == 0
        spx = 1; 
    else
        ni = length(x{i}); 
        if nargin <= 1
            spx = spx + (sqrt(ni)-norm(x{i},1)/norm(x{i},2))/(sqrt(ni)-1);
        else
            nw = norm(w{i},2); 
            spx = spx + (nw-w{i}'*abs(x{i})/norm(x{i},2))/(nw-min(w{i}));
        end
    end
end
spx = spx/r; 