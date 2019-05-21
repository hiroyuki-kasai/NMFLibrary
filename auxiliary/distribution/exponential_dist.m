function E = exponential_dist()

    
    E.exponential_draw = @exponential_draw;
    function exp_dist = exponential_draw(lambdax)
        
        scale = 1.0 / lambdax;
        
        exp_dist = rand(scale);

    end

end

