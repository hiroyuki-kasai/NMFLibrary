% Nonnegativity Constrained Least Squares with Multiple Righthand Sides 
%      using Active Set method
%
% This software solves the following problem: given A and B, find X such that
%            minimize || AX-B ||_F^2 where X>=0 elementwise.
%
% Reference:
%      Charles L. Lawson and Richard J. Hanson, Solving Least Squares Problems, 
%            Society for Industrial and Applied Mathematics, 1995
%      M. H. Van Benthem and M. R. Keenan, 
%            Fast Algorithm for the Solution of Large-scale Non-negativity-constrained Least Squares Problems,
%            J. Chemometrics 2004; 18: 441-450
%
% Written by Jingu Kim (jingu.kim@gmail.com)
%               School of Computational Science and Engineering,
%               Georgia Institute of Technology
%
% Please send bug reports, comments, or questions to Jingu Kim.
%
% Updated Feb-20-2010
% Updated Mar-20-2011: numChol, numEq
%
% <Inputs>
%        A : input matrix (m x n) (by default), or A'*A (n x n) if isInputProd==1
%        B : input matrix (m x k) (by default), or A'*B (n x k) if isInputProd==1
%        overwrite : (optional, default:0) if turned on, unconstrained least squares solution is computed in the beginning
%        isInputProd : (optional, default:0) if turned on, use (A'*A,A'*B) as input instead of (A,B)
%        init : (optional) initial value for X
% <Outputs>
%        X : the solution (n x k)
%        Y : A'*A*X - A'*B where X is the solution (n x k)
%        iter : number of systems of linear equations solved
%        success : 0 for success, 1 for failure.
%                  Failure could only happen on a numericall very ill-conditioned problem.

function [ X,Y,success,numChol,numEq ] = nnlsm_activeset( A, B, overwrite, isInputProd, init)
    if nargin<3, overwrite=0;, end
    if nargin<4, isInputProd=0;, end
    
    if isInputProd
        AtA=A;,AtB=B;
    else
        AtA=A'*A;, AtB=A'*B;
    end

    if size(AtA,1)==1
        X = AtB/AtA; X(X<0) = 0;
        Y = AtA*X - AtB;
        numChol = 1; numEq = size(AtB,2); success = 1;
        return
    end
        
    [n,k]=size(AtB);
    MAX_ITER = n*5;
    % set initial feasible solution
    if overwrite
        [X,numChol,numEq] = normalEqComb(AtA,AtB);
        PassSet = (X > 0);
        NotOptSet = any(X<0);
    elseif nargin>=5
        X = init;
        X(X<0)=0;
        PassSet = (X > 0);
        NotOptSet = true(1,k);
        numChol = 0;
        numEq = 0;
    else
        X = zeros(n,k);
        PassSet = false(n,k);
        NotOptSet = true(1,k);
        numChol = 0;
        numEq = 0;
    end
    
    Y = zeros(n,k);
    Y(:,~NotOptSet)=AtA*X(:,~NotOptSet) - AtB(:,~NotOptSet);
    NotOptCols = find(NotOptSet);
    
    bigIter = 0;, success=0;
    while(~isempty(NotOptCols))
        bigIter = bigIter+1;
        if ((MAX_ITER >0) && (bigIter > MAX_ITER))   % set max_iter for ill-conditioned (numerically unstable) case
            success = 1;, break
        end
        
        % find unconstrained LS solution for the passive set
        [ Z,tempChol,tempEq ] = normalEqComb(AtA,AtB(:,NotOptCols),PassSet(:,NotOptCols));
        numChol = numChol + tempChol;
        numEq = numEq + tempEq;

        Z(abs(Z)<1e-12) = 0;                 % One can uncomment this line for numerical stability.

        InfeaSubSet = Z < 0;
        InfeaSubCols = find(any(InfeaSubSet));
        FeaSubCols = find(all(~InfeaSubSet));
        
        if ~isempty(InfeaSubCols)               % for infeasible cols
            ZInfea = Z(:,InfeaSubCols);
            InfeaCols = NotOptCols(InfeaSubCols);
            Alpha = zeros(n,length(InfeaSubCols));, Alpha(:) = Inf;
            [i,j] = find(InfeaSubSet(:,InfeaSubCols));
            InfeaSubIx = sub2ind(size(Alpha),i,j);
            if length(InfeaCols) == 1
                InfeaIx = sub2ind([n,k],i,InfeaCols * ones(length(j),1));
            else
                InfeaIx = sub2ind([n,k],i,InfeaCols(j)');
            end
            Alpha(InfeaSubIx) = X(InfeaIx)./(X(InfeaIx)-ZInfea(InfeaSubIx));

            [minVal,minIx] = min(Alpha);
            Alpha(:,:) = repmat(minVal,n,1);
            X(:,InfeaCols) = X(:,InfeaCols)+Alpha.*(ZInfea-X(:,InfeaCols));
            IxToActive = sub2ind([n,k],minIx,InfeaCols);
            X(IxToActive) = 0;
            PassSet(IxToActive) = false;
        end
        if ~isempty(FeaSubCols)                 % for feasible cols
            FeaCols = NotOptCols(FeaSubCols);
            X(:,FeaCols) = Z(:,FeaSubCols);
            Y(:,FeaCols) = AtA * X(:,FeaCols) - AtB(:,FeaCols);

            Y( abs(Y)<1e-12 ) = 0;               % One can uncomment this line for numerical stability.
            
            NotOptSubSet = (Y(:,FeaCols) < 0) & ~PassSet(:,FeaCols);
            NewOptCols = FeaCols(all(~NotOptSubSet));
            UpdateNotOptCols = FeaCols(any(NotOptSubSet));
            if ~isempty(UpdateNotOptCols)
                [minVal,minIx] = min(Y(:,UpdateNotOptCols).*~PassSet(:,UpdateNotOptCols));
                PassSet(sub2ind([n,k],minIx,UpdateNotOptCols)) = true;
            end
            NotOptSet(NewOptCols) = false;
            NotOptCols = find(NotOptSet);
        end
    end
end
