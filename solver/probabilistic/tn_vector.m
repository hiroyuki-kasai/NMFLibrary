function T = tn_vector()

    T.expectation = @TN_vector_expectation;
    function [exp] = TN_vector_expectation(mus, taus)

        sigmas = 1.0 ./ sqrt(taus);
        x = - mus ./ sigmas;
        %lambdax = norm.pdf(x)/(0.5*erfc(x/sqrt(2)));
        lambdax = pdf('Normal',x,0,1) ./ (0.5 * erfc(x/sqrt(2)));
        exp = mus + sigmas .* lambdax;

        for i = 1 : length(exp)
            v = exp(i);
            mu = mus(i);
            tau = taus(i);
            sigma = sigmas(i);

            if mu < -30 * sigma
                v = 1/(abs(mu)*tau);
            else
                % do nothing;
            end

            if isnan(v)
                v = 0;
            end

            exp(i) = v;
        end


        exp(isinf(exp)) = 0;

    end       


    T.variance = @TN_vector_variance;
    function [var] = TN_vector_variance(mus, taus)

        sigmas = 1.0 ./ sqrt(taus);
        x = - mus ./ sigmas;
        %lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)));
        lambdax = pdf('Normal',x,0,1) ./ (0.5 * erfc(x/sqrt(2)));    
        deltax = lambdax .* (lambdax-x);
        var = sigmas.^2 .* ( 1 - deltax );

        for i = 1 : length(var)
            v = var(i);
            mu = mus(i);
            tau = taus(i);
            sigma = sigmas(i);

            if mu < -30 * sigma
                v = (1/(abs(mu)*tau)).^2;
            else
                % do nothing;
            end

            if isnan(v)
                v = 0;
            end

            var(i) = v;
        end

        var(isinf(var)) = 0;
    end
end

