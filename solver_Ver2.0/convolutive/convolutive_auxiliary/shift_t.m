function [H_t] = shift_t(H,t)

% Shifts the matrix H by t spots to the right
% Use a negative t to shift to the left
% To be used with the convolutive NMF algorithms

[K,N] = size(H);

if t>0
      H_trunc = H(:,1:N-t);
      H_t = [zeros(K,t)+eps,H_trunc];
    
elseif t<0
      H_trunc = H(:,-t+1:end);
      H_t = [H_trunc,zeros(K,-t)+eps];
else
      H_t = H;
end

end