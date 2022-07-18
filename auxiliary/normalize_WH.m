function [W_new, H_new] = normalize_WH(V, W, H, rank, type)

    switch type
        case 'type1'
            % scale initialization so that argmin_a ||a * WH - X||_F = 1   
            
            VHt = V * H'; 
            HHt = H * H'; 
        
            scaling = sum(sum(VHt.*W))/sum(sum(HHt.*(W'*W))); 
            W = W * scaling; 
            
            % scale W and H so that columns/rows have the same norm, that is, 
            % ||W(:,k)|| = ||H(k,:)|| for all k. 
            normW = sqrt((sum(W.^2)))+1e-16;
            normH = sqrt((sum(H'.^2)))+1e-16;
            for k = 1 : rank
                W(:,k) = W(:,k)/sqrt(normW(k))*sqrt(normH(k));
                d(k) = sqrt(normW(k))/sqrt(normH(k)); 
                H(k,:) = H(k,:)*d(k);
            end 
            
            W_new = W;
            H_new = H;
            
        case 'type2'            
            
            % normalizes rows in R so that the product LR stays the same
            % handle_zeros option: leaves rows that sum up to 0 as they were

            coeff_h = sum(H, 2);

            % added by HK
            coeff_h = max(coeff_h, 1e-16);

            coeff_w = 1 ./ coeff_h;
            left = diag(coeff_h);
            right = diag(coeff_w);
            W_new = W * left;
            H_new = right * H; 


        case 'type3'    
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %
            % H^Te <= e entries in cols of H sum to at most 1
            Hn = SimplexProj( H );
            if norm(Hn - H) > 1e-3*norm(Hn); 
               H = Hn; 
               % reoptimize W, because this normalization is NOT w.l.o.g. 
               options.inneriter = 100; 
               options.H = W'; 
               %W = nnls_PFGM(X',H',options);  % HK
               W = nnls_fpgm(V',H',options);   % HK
               
               W = W'; 
            end
            H = Hn; 

            W_new = W;
            H_new = H;            


        case 'type4' 
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %            
            % He = e, entries in rows of H sum to 1

            scalH = sum(H');
            H = diag( scalH.^(-1) )*H;
            W = W*diag( scalH );

            W_new = W;
            H_new = H;   

        case 'type5'            
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %
            % W^T e = e, entries in cols of W sum to 1

            scalW = sum(W);
            H = diag( scalW )*H;
            W = W*diag( scalW.^(-1) );

            W_new = W;
            H_new = H;      
            
        otherwise 

    end

end
        