function new_alpha = update_alpha(alpha, q)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % new_alpha = update_alpha(alpha, q)                       %
    %                                                          %
    % Updates parameter alpha: see Nesterov book, page 90      %
    %                                                          %
    % A. P. Liavas, May 13, 2015                               %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    a = 1;
    b = alpha^2 - q;
    c = -alpha^2;
    D = b^2 - 4 * a * c;

    new_alpha = (-b+sqrt(D))/2;
end

