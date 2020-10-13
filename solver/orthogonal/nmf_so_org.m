function [C_best, S_best, obj_best, orth_best] = nmf_so(X, K, opts)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NMF (Nonnegative Matrix Factorization) with soft orthogonal constraint
%
%                       (c) Motoki Shiga, Gifu University, Japan
%
%-- INPUT --------------------------------------------------------------
%
%   X    : matrix with the size of (Nxy x Nch)
%          Nxy: the number of measurement points on specimen
%          Nch: the number of spectrum channels
%   K    : the number of components
%   wo   : weight for the orthologal constraint on G
%   opts : options
%
%-- OUTPUT -------------------------------------------------------------
%
%   C_best   : densities of components at each point
%   S_best   : spectrums of components
%   obj_best : learning curve (error value after each update)
%   
%
%  Reference
%  [1] Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda, Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji, 
%      "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
%      Ultramicroscopy, Vol.170, p.43-59, 2016.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options
reps   = opts.reps;      % the number of initializations
itrMax = opts.itrMax;    % the maximum number of updates
wo     = opts.wo;        % weight of orthogonal constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Nxy, Nch] = size(X);
obj_best = inf;
disp(' ')
disp('Running NMF with Soft Orthogonal constraint....')
disp('Optimizing components from different initializations:')
for rep = 1:reps
  disp( strcat([num2str(rep),' / ',num2str(reps)]) )
  % Initialization of matrix C
  C = rand(Nxy,K);  
  for j = 1:K
    C(:,j) = C(:,j) / (C(:,j)'*C(:,j)); % normalization
  end
  % Initialization of matrix S
  i = randsample(Nxy,K);
  S = X(i,:)';
  
  %update C and S by HALS algorithm
  [C,S,obj,orth] = nmfso_hals(X,C,S,K,wo,itrMax);

  %choose the best optimization result
  if obj(end) < obj_best(end)
    C_best   = C;  
    S_best   = S;  
    obj_best = obj;
    orth_best = orth;
  end
end

%remove small values
C_best(C_best<eps) = 0;   S_best(S_best<eps) = 0;

%sort component by the order of spectral peak positions 
[~,k]=max(C_best);     [~,k]=sort(k);
C_best = C_best(:,k);  S_best = S_best(:,k);

disp('Finish the optimization of our model!')
disp(' ')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization from a initialization by algorithm HALS 
function [C,S,obj,orth] = nmfso_hals(X,C,S,K,wo,itrMax)
N = numel(X);

cj = sum(C,2);
obj = nan(itrMax,1);
orth = nan(itrMax,1);
for itr = 1:itrMax

  %update C
  XS = X*S;   S2 = S'*S;
  for j = 1:K
    cj     = cj - C(:,j);
    C(:,j) = XS(:,j) - C*S2(:,j) + S2(j,j)*C(:,j);
    C(:,j) = C(:,j) - wo*(cj'*C(:,j))/(cj'*cj)*cj;
    %replace negative values with zeros (tiny positive velues)
    C(:,j) = max( (C(:,j) + abs(C(:,j)))/2, eps);
    C(:,j) = C(:,j) / sqrt(C(:,j)'*C(:,j)); %normalize
    cj     = cj + C(:,j);
  end
  
  %update S
  XC = X'*C;   C2 = C'*C;
  for j = 1:K
    S(:,j) = XC(:,j) - S*C2(:,j) + C2(j,j)*S(:,j);
    %replace negative values with zeros (tiny positive velues)
    S(:,j) = max( (S(:,j) + abs(S(:,j)))/2, eps);
  end
  
  %cost function
  X_est = C*S';  %reconstracted data matrix
  obj(itr) = sum(sum((X-X_est).^2))/2;  %MSE
  orth(itr) = norm(C'*C - eye(K),'fro');
  
  %cheack convergence
  if (itr>1) && (abs(obj(itr-1) - obj(itr)) < eps)
    obj = obj(1:itr);
    break;
  end
end

end
