function [V, Ro] = add_outlier(rho, F, N, Vo, outlier_min, outlier_max, val_min, val_max)
% Function to add outlier to input data
%
% This file is part of NMFLibrary.
%
% Created by H.Kasai
%


    if nargin < 5 
        outlier_min = 30;
    end
    if nargin < 6 
        outlier_max = 50;
    end
    if nargin < 7
        val_min = 0;
    end
    if nargin < 8
        val_max = 50;
    end
    
    dense = rho * F;
    Ro = zeros(F, N);
    
    for i = 1 : N
        n_before = 0;

            for f = 1 : dense
                c = randi(F);
                 Ro(c,i) =  randi([outlier_min, outlier_max]);
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

