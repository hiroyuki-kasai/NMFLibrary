function new_inner_max_epoch = change_inner_max_epoch(X, W, options) 
    
    new_inner_max_epoch = options.inner_max_epoch;

    if isfield(options,'inner_max_epoch_parameter')
        [m, n] = size(X); 
        [m, r] = size(W); 
        alpha_inner_max_epoch = 1 + ceil( options.inner_max_epoch_parameter * (nnz(X)+m*r)/(n*r) ); 
        new_inner_max_epoch = min(alpha_inner_max_epoch, options.inner_max_epoch); 
    end
end
