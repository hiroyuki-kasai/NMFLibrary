% Given a set of nonnegative vectors x{i}, i=1,2,...,r 
% 
% Compute the function g(mu) = sum_i beta_i [ e^T x_i(mu) ] 
% where 
% beta_i = 1/[ ||w{i}|| - min_j w{i}(j) ], and 
% If x{i} - beta_i mu w{i} contains at least one postive entry: 
%      x_i(mu) = (x{i} - beta_i mu w{i})_+; 
%      x_i(mu) = x_i(mu)/||x_i(mu)||_2; 
% Otherwise 
%     x_i(mu) is 1-sparse with nonzero entry equal to one at position
%     corresponding to the largest entry of x{i} - beta_i mu w{i} 

function [vgmu,xp,gradg] = wgmu(x,w,mu); 

vgmu = 0; 
gradg = 0; 
for i = 1 : length(x)
    ni = length(x{i}); 
    betai = 1/(norm(w{i})-min(w{i})); 
    xp{i} = x{i} - mu*betai*w{i}; 
    indtp = find(xp{i} > 0); 
    % Gradient g(mu) 
    if ~isempty(indtp)
        xp{i} = max(0,xp{i});  
        f2 = norm(xp{i});
        if nargout >= 3
            nip = w{i}(indtp)'*w{i}(indtp); 
            
            gradg = gradg + betai^2 * ( - nip * f2^(-1) ... 
                + (w{i}(indtp)'*xp{i}(indtp))^2 * f2^(-3)  );
        end
        xp{i} = xp{i}/norm(xp{i},2); 
        vgmu = vgmu + betai*sum(xp{i}.*w{i}); 
    else
        [~,im] = max(xp{i}); 
        xp{i} = zeros( ni , 1 ); 
        xp{i}(im) = 1; 
        vgmu = vgmu + betai*w{i}(im); 
    end
end