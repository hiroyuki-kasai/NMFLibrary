function [Vo, Ro, No, V] = generate_syntheticdata(F, N, K, noise_level, rho)
% 
%     % Vo
%     W = randn(F,K);
%     W = max(W, 0);
%     W = min(W, 1); 
%     
%     H = randn(K,N);
%     H = max(H, 0);
%     H = min(H, 1); 
%     
%     Vo = W * H;
%     
%     % noise
     No = noise_level*randn(F,N);
%     
%     % add outlier
%     [V, Ro] = add_outlier(rho, F, N, Vo);
%     
%     % V
%     V = V + No;
%     %V = max(V, 0);
%     %V = min(V,1);     

    %
    sigma2 = 1 / sqrt(K);
    HN = makedist('Normal', 'mu', 0, 'sigma', sqrt(sigma2));
    Vo_n = random(HN, F, N);
    Vo = abs(Vo_n) ;
    Vo = min(Vo, 1);

    nu = rho;
    nu_tilda = 0.1;
    I = nu * N;
    card = nu_tilda * F;
    Ro = zeros(F,N);
    if rho > 0
        for i = 1 : N
            n_before = 0;
            if i < I
                for f = 1 : card
                    c = randi(F);
                     Ro(c,i) =  1 + (1+1)*rand(1, 1);
                     n = nnz(Ro(:,i));
                     if n_before == n
                         f = f - 1;
                     end
                     n_before = n;
                end               

            end
        end
    end
    
    % V
    V = Vo + Ro + No;
    %V = max(V, 0);
    %V = min(V,1); 
    
    index = find(V<0);
    V(index) = 0;

end

