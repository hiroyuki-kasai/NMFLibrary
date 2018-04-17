function [V, Ro] = add_outlier_new(rho, F, N, Vo, outlier_min, outlier_max, val_min, val_max)

    dense = rho * F;
    Ro = zeros(F, N);
    
    for i = 1 : N
        n_before = 0;

            for f = 1 : dense
                c = randi(F);
                 Ro(c,i) =  randi([outlier_min, outlier_max])/10;
                 n = nnz(Ro(:,i));
                 if n_before == n
                     f = f - 1;
                 end
                 n_before = n;
            end
    end
    V = Vo + Ro ;
    V = max(V, val_min);
    V = min(V, val_max);
end

