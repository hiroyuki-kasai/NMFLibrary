function psnr = calc_psnr(V, W, H)

    % PSNR = 10 log10 (MAX^2/MSE)
    %
    %       MAX_VAL: Maximum value of pixels
    
    max_val = max(max(V));
    mse = calc_mse(V, W, H);
    psnr = 10 * log10 (max_val.^2/mse);    
    
end
