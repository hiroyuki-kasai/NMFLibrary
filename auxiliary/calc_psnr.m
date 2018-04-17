function psnr = calc_psnr(Vo, W, H)

    % PSNR = 10 log10 (MAX^2/MSE)
    %
    %       MAX_VAL: Maximum value of pixels
    
    max_val = max(max(Vo));
    mse = calc_mse(Vo, W, H);
    psnr = 10 * log10 (max_val.^2/mse);    
    
end
