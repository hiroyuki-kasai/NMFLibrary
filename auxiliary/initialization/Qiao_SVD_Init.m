% New SVD based initialization strategy for Non-negative Matrix Factorization
% Hanli Qiao


function [W, H] = Qiao_SVD_Init(Z, rank)
    [u, s, v, p] = ChoosingR(Z);
    W = abs(u(:,1:rank));
    H = abs(s(1:rank,:)*v');    
end

function [u, s, v, p] = ChoosingR(Z)
    [u,s,v] = svd(Z);
    sum1= sum(s);
    sum2=sum(sum1);
    extract=0;
    p = 0;
    dsum=0;
    while(extract/sum2<0.90)
        p = p + 1;
        dsum=dsum+s(p,p);
        extract=dsum;
    end
end

