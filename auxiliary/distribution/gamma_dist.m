function D = gamma_dist()

    
    D.gamma_expectation = @gamma_expectation;
    function ratio = gamma_expectation(alpha, beta)
        
        ratio = alpha ./ beta;

    end


    D.gamma_expectation_log = @gamma_expectation_log;
    function val = gamma_expectation_log(alpha, beta)
        
        val = psi(alpha) - log(beta);

    end


end

