function mse = calc_mse(Vo, W, H)
    
    F = size(Vo, 1);
    N = size(Vo, 2);

    mse = norm(Vo - W * H,'fro')^2 / (F*N);

end
