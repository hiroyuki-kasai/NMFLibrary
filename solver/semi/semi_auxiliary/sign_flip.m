function [sgns,loads] = sign_flip(loads,X)

% [sgns,loads] = sign_flip(loads,X)
% loads is a cell of loadings
% X     is the data array
% sgns is a MxF matrix where sgns(m,f) is the sign of loading f in mode m
%
% If using svd ([u,s,v]=svd(X)) then loads{1}=u*s; and loads{2}=v;
% If using an F-component PCA model ([t,p]=pca(X,F), then loads{1}=t; and
% loads{2}=p;
% 
% Copyright 2007 R. Bro, E. Acar, T. Kolda - www.models.life.ku.dk

% for PARAFAC and two-way

if isa(X,'dataset')
  inc = X.includ;            
    X = X.data(inc{:});
end

order = length(size(X));
for i=1:order
  F(i) = size(loads{i},2);
end

for m = 1:order % for each mode
    for f=1:F(m) % for each component
      s=[];
      a = loads{m}(:,f);
      a = a /(a'*a);
      x = subtract_otherfactors(X, loads, m, f);
      for i=1:size(x(:,:),2) % for each column
         s(i)=(a'*x(:,i));
         s(i)=sign(s(i))*power(s(i),2);
      end
      S(m,f) =sum(s);
    end
end
sgns = sign(S);

for f=1:F(1) %each component
  for i=1:size(sgns,1) %each mode
    se = length(find(sgns(:,f)==-1));
    if (rem(se,2)==0 )
        loads{i}(:,f)=sgns(i,f)*loads{i}(:,f);
    else
        % disp('Odd number of negatives!')
        sgns(:,f) = handle_oddnumbers(S(:,f));
        se = length(find(sgns(:,f)==-1));
        if (rem(se,2)==0)
            loads{i}(:,f)=sgns(i,f)*loads{i}(:,f);
        else
            disp('Something Wrong!!!')
        end
    end
  end  %each mode
end %each component

%----------------------------------------------------------------------
function sgns=handle_oddnumbers(Bcon)

sgns=sign(Bcon);
nb_neg=find(Bcon<0);
[min_val, index]=min(abs(Bcon));
if (Bcon(index)<0)
    sgns(index)=-sgns(index);
% since this function is called nb_neg should be greater than 0, anyway
elseif ((Bcon(index)>0) && (nb_neg>0))
    sgns(index)=-sgns(index);
end


%------------------------------------------------------------------------
function x = subtract_otherfactors(X, loads, mode, factor)

order=length(size(X));
x = permute(X,[mode 1:mode-1 mode+1:order]);
loads = loads([mode 1:mode-1 mode+1:order]);

for m = 1: order
   loads{m}=loads{m}(:, [factor 1:factor-1 factor+1:size(loads{m},2)]); 
   L{m} = loads{m}(:,2:end);
end
M = outerm(L);
x=x-M;




function mwa = outerm(facts,lo,vect)

if nargin < 2
  lo = 0;
end
if nargin < 3
  vect = 0;
end
order = length(facts);
if lo == 0
  mwasize = zeros(1,order);
else
  mwasize = zeros(1,order-1);
end
k = 0;
for i = 1:order
  if i ~= lo
    [m,n] = size(facts{i});
    k = k + 1;
    mwasize(k) = m;
    if k > 1
    else
      nofac = n;
    end
  end
end
mwa = zeros(prod(mwasize),nofac);

for j = 1:nofac
  if lo ~= 1
    mwvect = facts{1}(:,j);
    for i = 2:order
	  if lo ~= i
		mwvect = mwvect*facts{i}(:,j)';
		mwvect = mwvect(:);
	  end
    end
  elseif lo == 1
    mwvect = facts{2}(:,j);
	for i = 3:order
	  mwvect = mwvect*facts{i}(:,j)';
	  mwvect = mwvect(:);
	end
  end
  mwa(:,j) = mwvect;
end
% If vect isn't one, sum up the results of the factors and reshape
if vect ~= 1
  mwa = sum(mwa,2);
  mwa = reshape(mwa,mwasize);
end