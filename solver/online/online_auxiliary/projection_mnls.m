function A = projection_mnls(X, B, A)
    
    s = svd(B'*B);
    L = max(s);
    %eigen_values = eig(B'*B);
    %L = max(eigen_values);
    
    grad = - X * B + A*(B' * B);
    A = A - 1/L * grad;
    A = max(A, 0); 
end


