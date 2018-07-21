function [ Z,numChol,numEq ] = normalEqComb( AtA,AtB,PassSet )
% Solve normal equations using combinatorial grouping.
% Reference:
%        M. H. Van Benthem and M. R. Keenan,
%        Fast Algorithm for the Solution of Large-scale Non-negativity-constrained
%        Least Squares Problems.
%        J. Chemometrics, 18, pp. 441-450, 2004. 
%
% This function was originally adopted from above paper, but a few
% important modifications have been made.
%
% Modified by Jingu Kim (jingu.kim@gmail.com)
%             School of Computational Science and Engineering,
%             Georgia Institute of Technology
%
% Updated Aug-12-2009
% Updated Mar-13-2011: numEq,numChol
%
% numChol : number of unique cholesky decompositions done
% numEqs : number of systems of linear equations solved

    if isempty(AtB)
        Z = [];
        numChol = 0; numEq = 0;
    elseif (nargin==2) || all(PassSet(:))
        Z = AtA\AtB;
        numChol = 1; numEq = size(AtB,2);
    elseif size(AtA,1) ==1
        Z = AtB/AtA;
        numChol = 1; numEq = size(AtB,2);
    else
        Z = zeros(size(AtB));
        [n,k1] = size(PassSet);
        if k1==1 % Treat a case with a single righthand side seperately
            if any(PassSet)>0
                Z(PassSet)=AtA(PassSet,PassSet)\AtB(PassSet); 
                numChol = 1; numEq = 1;
            else
                numChol = 0; numEq = 0;
            end
        else
            % Original function has limitations in the length of solution vector.
            [sortedPassSet,sortIx] = sortrows(PassSet');
            breaks = any(diff(sortedPassSet)');
            breakIx = [0 find(breaks) k1];
            % Skip columns with no passive sets
            if any(sortedPassSet(1,:))==0;
                startIx = 2;
            else
                startIx = 1;
            end
            numChol = 0; 
            numEq = k1-breakIx(startIx);
            for k=startIx:length(breakIx)-1
                cols = sortIx(breakIx(k)+1:breakIx(k+1));
                vars = sortedPassSet(breakIx(k)+1,:)';
                Z(vars,cols) = AtA(vars,vars)\AtB(vars,cols);
                numChol = numChol + 1;
            end
        end
    end
end
