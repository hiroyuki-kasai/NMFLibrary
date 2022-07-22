% Nonnegativity Constrained Least Squares with Multiple Righthand Sides 
%      using Block Principal Pivoting method
%
% This software solves the following problem: given A and B, find X such that
%              minimize || AX-B ||_F^2 where X>=0 elementwise.
%
% Reference:
%      Jingu Kim and Haesun Park. Fast Nonnegative Matrix Factorization: An Activeset-like Method and Comparisons,
%      SIAM Journal on Scientific Computing, 33(6), pp. 3261-3281, 2011.
%
% Written by Jingu Kim (jingu.kim@gmail.com)
%            School of Computational Science and Engineering,
%            Georgia Institute of Technology
%
% Note that this algorithm assumes that the input matrix A has full column rank.
% This code comes with no guarantee or warranty of any kind. 
% Please send bug reports, comments, or questions to Jingu Kim.
%
% Modified Feb-20-2009
% Modified Mar-13-2011: numChol and numEq
%
% <Inputs>
%        A : input matrix (m x n) (by default), or A'*A (n x n) if isInputProd==1
%        B : input matrix (m x k) (by default), or A'*B (n x k) if isInputProd==1
%        isInputProd : (optional, default:0) if turned on, use (A'*A,A'*B) as input instead of (A,B)
%        init : (optional) initial value for X
% <Outputs>
%        X : the solution (n x k)
%        Y : A'*A*X - A'*B where X is the solution (n x k)
%        success : 0 for success, 1 for failure.
%                  Failure could only happen on a numericall very ill-conditioned problem.
%        numChol : number of unique cholesky decompositions done
%        numEqs : number of systems of linear equations solved

function [ X,Y,success,numChol,numEq ] = nnlsm_blockpivot( A, B, isInputProd, init )
    if nargin<3, isInputProd=0;, end
    if isInputProd
        AtA = A;, AtB = B;
    else
        AtA = A'*A;, AtB = A'*B;
    end

    if size(AtA,1)==1
        X = AtB/AtA; X(X<0) = 0;
        Y = AtA*X - AtB;
        numChol = 1; numEq = size(AtB,2); success = 1;
        return
    end
    
    [n,k]=size(AtB);
    MAX_BIG_ITER = n*5;
    % set initial feasible solution
    X = zeros(n,k);
    if nargin<4
        Y = - AtB;
        PassiveSet = false(n,k);
        numChol = 0;
        numEq = 0;
    else
        PassiveSet = (init > 0);
        [ X,numChol,numEq] = normalEqComb(AtA,AtB,PassiveSet);
        Y = AtA * X - AtB;
    end
    % parameters
    pbar = 3;
    P = zeros(1,k);, P(:) = pbar;
    Ninf = zeros(1,k);, Ninf(:) = n+1;

    NonOptSet = (Y < 0) & ~PassiveSet;
    InfeaSet = (X < 0) & PassiveSet;
    NotGood = sum(NonOptSet)+sum(InfeaSet);
    NotOptCols = NotGood > 0;

    bigIter = 0;, success=0;
    while(~isempty(find(NotOptCols)))
        bigIter = bigIter+1;
        if ((MAX_BIG_ITER >0) && (bigIter > MAX_BIG_ITER))   % set max_iter for ill-conditioned (numerically unstable) case
            success = 1;, break
        end

        Cols1 = NotOptCols & (NotGood < Ninf);
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1);
        Cols3Ix = find(NotOptCols & ~Cols1 & ~Cols2);
        if ~isempty(find(Cols1))
            P(Cols1) = pbar;,Ninf(Cols1) = NotGood(Cols1);
            PassiveSet(NonOptSet & repmat(Cols1,n,1)) = true;
            PassiveSet(InfeaSet & repmat(Cols1,n,1)) = false;
        end
        if ~isempty(find(Cols2))
            P(Cols2) = P(Cols2)-1;
            PassiveSet(NonOptSet & repmat(Cols2,n,1)) = true;
            PassiveSet(InfeaSet & repmat(Cols2,n,1)) = false;
        end
        if ~isempty(Cols3Ix)
            for i=1:length(Cols3Ix)
                Ix = Cols3Ix(i);
                toChange = max(find( NonOptSet(:,Ix)|InfeaSet(:,Ix) ));
                if PassiveSet(toChange,Ix)
                    PassiveSet(toChange,Ix)=false;
                else
                    PassiveSet(toChange,Ix)=true;
                end
            end
        end
        [ X(:,NotOptCols),tempChol,tempEq ] = normalEqComb(AtA,AtB(:,NotOptCols),PassiveSet(:,NotOptCols));
        numChol = numChol + tempChol;
        numEq = numEq + tempEq;
        X(abs(X)<1e-12) = 0;            % One can uncomment this line for numerical stability.
        Y(:,NotOptCols) = AtA * X(:,NotOptCols) - AtB(:,NotOptCols);
        Y(abs(Y)<1e-12) = 0;            % One can uncomment this line for numerical stability.
        
        % check optimality
        NotOptMask = repmat(NotOptCols,n,1);
        NonOptSet = NotOptMask & (Y < 0) & ~PassiveSet;
        InfeaSet = NotOptMask & (X < 0) & PassiveSet;
        NotGood = sum(NonOptSet)+sum(InfeaSet);
        NotOptCols = NotGood > 0;
    end
end
