function mse = calc_mse(V, W, H)
    
    [m, n] = size(V);

    f_val = nmf_cost(V, W, H, []);
    
    mse = 2 * f_val / (m*n);

end
