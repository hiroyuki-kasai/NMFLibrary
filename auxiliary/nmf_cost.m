function f = nmf_cost(V, W, H, R)

    f = norm(V - W * H - R,'fro')^2 / 2 ;
    
end
