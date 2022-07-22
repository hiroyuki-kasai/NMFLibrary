function [ x,y,success,iter ] = nnls1_asgivens( A,b,overwrite, isInputProd, init )
% Nonnegativity-constrained least squares for single righthand side : minimize |Ax-b|_2
% Jingu Kim (jingu.kim@gmail.com)
%
% Reference:
%      Jingu Kim and Haesun Park. Fast Nonnegative Matrix Factorization: An Activeset-like Method and Comparisons,
%      SIAM Journal on Scientific Computing, 33(6), pp. 3261-3281, 2011.
%
% Updated 2011.03.20: First implemented, overwrite option
% Updated 2011.03.21: init option
% Updated 2011.03.23: Givens updating not always

    if nargin<3, overwrite = false; end
    if nargin<4, isInputProd = false; end

    if isInputProd
        AtA=A;,Atb=b;
    else
        AtA=A'*A;, Atb=A'*b;
    end
    n=size(Atb,1);
    MAX_ITER = n*5;

    % set initial feasible solution
    if overwrite
        x = AtA\Atb;
        x(x<0) = 0;
        PassiveList = find(x > 0)';
        R = chol(AtA(PassiveList,PassiveList));
        Rinv_b = (R')\Atb(PassiveList);
        iter = 1;
    else
        if nargin<5
            PassiveList = [];
            R = [];
            Rinv_b = zeros(0,0);
            x = zeros(n,1);
        else
            x = init;
            x(x<0) = 0;
            PassiveList = find(x > 0)';
            R = chol(AtA(PassiveList,PassiveList));
            Rinv_b = (R')\Atb(PassiveList);
        end
        iter=0;
    end

    success=1;
    while(success)
        if iter >= MAX_ITER, break, end
        % find unconstrained LS solution for the passive set
        if ~isempty(PassiveList)
            z = R\Rinv_b;
            iter = iter + 1;
        else
            z = [];
        end
        z( abs(z)<1e-12 ) = 0;                  % One can uncomment this line for numerical stability.

        InfeaSet = find(z < 0);
        if isempty(InfeaSet)                    % if feasibile
            x(:) = 0;
            x(PassiveList) = z;
            y = AtA * x - Atb;
            y( PassiveList) = 0;
            y( abs(y)<1e-12 ) = 0;              % One can uncomment this line for numerical stability.

            NonOptSet = find(y < 0);
            if isempty(NonOptSet), success=0;   % check optimality
            else
                [minVal,minIx] = min(y);
                PassiveList = [PassiveList minIx];  % increase passive set
                [R,Rinv_b] = cholAdd(R,AtA(PassiveList,minIx),Rinv_b,Atb(minIx));
            end
        else                                    % if not feasibile
            x_pass = x(PassiveList);
            x_infeaset = x_pass(InfeaSet);
            z_infeaset = z(InfeaSet);
            [minVal,minIx] = min(x_infeaset./(x_infeaset-z_infeaset));
            x_pass_new = x_pass+(z-x_pass).*minVal;
            x_pass_new(InfeaSet(minIx))=0;

            zeroSetSub = sort(find(x_pass_new==0),'descend');
            for i=1:length(zeroSetSub)
                subidx = zeroSetSub(i);
                PassiveList(subidx) = [];

                % Givens updating is not always better (maybe only in matlab?).
                if subidx >= 0.9 * size(R,2)
                    R = cholDelete(R,subidx);
                else
                    R = chol(AtA(PassiveList,PassiveList));
                end
            end
            Rinv_b = (R')\Atb(PassiveList);
            x_pass_new(x_pass_new == 0) = [];
            x(:) = 0;
            x(PassiveList) = x_pass_new;
        end
    end
end

function [new_R,new_d] = cholAdd(R,v,d,val)
    if isempty(R)
        new_R = sqrt(v);
        new_d = val/new_R;
    else
        n = size(R,1);
        new_R = zeros(n+1,n+1);
        new_R(1:n,1:n)=R;

        vec = zeros(n+1,1);
        vec(1:n)=R'\v(1:n);
        vec(n+1)=sqrt(v(n+1)-vec(1:n)'*vec(1:n));

        new_R(:,n+1) = vec;

        new_d = [d;zeros(1,1)];
        new_d(n+1) = (val-vec(1:n)'*d)/vec(n+1);
    end
end

function [new_R] = cholDelete(R,idx)
    n = size(R,1);
    new_R = R;
    new_R(:,idx) = [];

    for i=idx:n-1
        %G=getGivens(new_R(:,i),i,i+1);
        G=planerot(new_R([i i+1],i));
        new_R([i i+1],:)=G*new_R([i i+1],:);, new_R(i+1,i)=0;
    end
    new_R = new_R(1:n-1,1:n-1);
end

% function [G]=getGivens(a,i,j)
%     G=zeros(2,2);
%     [c,s]=givensRotation(a(i),a(j));
%     G(1,1)=c;
%     G(1,2)=s;
%     G(2,1)=-s;
%     G(2,2)=c;
% end
% 
% function [c,s]=givensRotation(a,b)
% % Givens Rotation to annihilate b with respect to a
%     if(b==0)
%         c=1;s=0;
%     else
%         if (abs(b)>abs(a))
%             t=-a/b;
%             s=1/sqrt(1+t*t);
%             c=s*t;
%         else
%             t=-b/a;
%             c=1/sqrt(1+t*t);
%             s=c*t;
%         end
%     end
% end
