function [ W,H ] = renormalize_convNMF( W,H )
% Renormalizes the latents factors of convolutive NMF
% so that the patches have a L1-norm equal to 1
% Author : Dylan Fagot

[M,K,T] = size(W);
[~,N] = size(H);

Lambda = zeros(1,K);
for kk=1:K
    patch_W_k = W(:,kk,:);
    Lambda(kk) = sum(patch_W_k(:));
end

for t=1:T
        W(:,:,t) = W(:,:,t)./repmat(Lambda,M,1);
end
    
H = repmat(Lambda',1,N).*H;

end