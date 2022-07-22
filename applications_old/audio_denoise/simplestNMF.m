function [D,X,Err] = simplestNMF(Y, opts)
    [n,T] = size(Y);
    
    if nargin<2
        opts = struct();
    end
    opts = initialize_opts(Y, opts);
    D = opts.D0;
    X = opts.X0;
    
    iter = 0;
    Err = zeros(opts.max_iter, 1);
    lambda_factor = repmat(opts.lambda.*ones(opts.m,1), 1, T);


    %% ----- Preprocess [start] -----
    nmflib_in_options.metric_type = opts.metric_type; 
    nmflib_in_options.verbose = opts.verbose;
    nmflib_in_options.x_init.W = D;
    nmflib_in_options.x_init.H = X; 
    nmflib_in_options.d_beta = opts.d_beta;
    [D, X, ~, nmflib_infos, nmflib_options] = pre_process('simplestNMF', Y, nmflib_in_options);
    % other necessary processes should be inserted here.
    %% ----- Preprocess [end] -----

    while iter < opts.max_iter
        iter = iter + 1;

        %update D
        if ~isempty(opts.updateD)
            %DX = D(:, opts.updateD)*X(opts.updateD, :);
            DX = D*X;
            if opts.d_beta<2
                DX(DX==0) = eps;
            end
            pgradD = (DX.^(opts.d_beta-1))*X(opts.updateD, :)';
            pgradD(pgradD==0)=eps;
            ngradD = ((DX.^(opts.d_beta-2)).*Y)*X(opts.updateD, :)';
            D(:, opts.updateD) = D(:, opts.updateD) .* ngradD ./ pgradD;
        end        
        
        %update X
        DX = D*X;
        if opts.d_beta<2
            DX(DX==0) = eps;
        end
        pgradX = D'*(DX.^(opts.d_beta-1));
        ngradX = D'*((DX.^(opts.d_beta-2)).*Y);
        X = X .* ngradX ./ (pgradX + lambda_factor);
        

        
        DX = D*X;
        DX(DX==0) = eps;
        Err(iter) = beta_divergence(Y, DX, opts.d_beta);
        %sparsy = sum(sum(lambda_factor.*X));
        delta = inf;
        if iter>1
            delta = (Err(iter-1)-Err(iter))/Err(iter-1);
        end


        %% ----- Innerprocess [start] -----
        nmflib_infos = inner_process('simplestNMF', Y, D, X, [], nmflib_options, nmflib_infos, iter);    
        %% ----- Innerprocess [end] -----  

        %fprintf('iter = %d, Err = %f, delta = %f, sparsy = %f\n', iter, Err(iter), delta, sparsy);
        %if delta < opts.conv_value
        %    break;
        %end
    end
    Err(iter+1:end) = []; 
end

function opts = initialize_opts(Y, opts)
    [n, T] = size(Y);
    if ~isfield(opts, 'm') opts.m = 1; end
    if ~isfield(opts, 'conv_value') opts.conv_value = 1e-3; end
    if ~isfield(opts, 'max_iter') opts.max_iter = 1000; end
    if ~isfield(opts, 'lambda') opts.lambda = eps; end
    if ~isfield(opts, 'beta') opts.d_beta = 2; end
    if ~isfield(opts, 'updateD') opts.updateD = 1:opts.m; end
    if ~isfield(opts, 'D0') opts.D0 = rand(n, opts.m); end
    if ~isfield(opts, 'X0') opts.X0 = rand(opts.m, T); end
end


function r = beta_divergence(A, B, beta)
    switch beta
        case 0
            r = sum(sum (A./B - log(A./B+eps) - 1));
        case 1
            r = sum(sum(A.*log(A./B+eps) - A + B));
        case 2
            r = .5 * norm(A - B, 'fro').^2;
    end
end