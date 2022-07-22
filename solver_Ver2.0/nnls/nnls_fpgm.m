function [H,WtW,WtX] = nnls_fpgm(X,W,options) 

% Computes an approximate solution of the following nonnegative least
% squares problem (NNLS)
%
%           min_{H >= 0} ||X-WH||_F^2
% 
% using a fast gradient method; 
% See Nesterov, Introductory Lectures on Convex Optimization: A Basic 
% Course, Kluwer Academic Publisher, 2004. 
% 
% Input / Output; see nnls_input_output.m  
% 
% + options.proj allows to use a contraints on the columns or rows of H so 
%   that the entries in each column/row sum to at most one 
%   options.proj = 0: no projection (default). 
%   options.proj = 1: projection of the columns on {x|x>=0, sum(x) <= 1} 
%   options.proj = 2: projection of the rows {x|x>=0, sum(x) = 1} 
%      
% + options.alpha0 is the FPGM  extrapolation parameter (default=0.05)
%
% Code modified from https://sites.google.com/site/nicolasgillis/code
%
% This file has been ported from 
%       nnls_FPGM.m at https://gitlab.com/ngillis/nmfbook/-/tree/master/algorithms
%       by Nicolas Gillis (nicolas.gillis@umons.ac.be)


    if nargin <= 2
        options = [];
    end
    if ~isfield(options,'delta')
        options.delta = 1e-6; % Stopping condition depending on evolution of the iterate V:
        % Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F
        % where V^{k} is the kth iterate.
    end
    if ~isfield(options,'inner_max_epoch')
        options.inner_max_epoch = 500; 
    end
    if ~isfield(options,'proj') % Projection on the unit simplex and the origin
        options.proj = 0; 
    end
    if ~isfield(options,'alpha0') % Parameter for FPGM ~ extrapolation parameter
        options.alpha0 = 0.05; 
    end

    W = full(W); 
    [m, n] = size(X);
    [m, r] = size(W);
    WtW = W'*W;
    WtX = W'*X;

    % If no initial matrices are provided, H is initialized as follows: 
    if ~isfield(options,'init') || isempty(options.init)
        H = nnls_init(X,W,WtW,WtX); 
    else
        H = options.init; 
    end

    % Hessian and Lipschitz constant 
    L = norm(WtW,2);  
    % Linear term 
    WtX = W'*X; 
    alpha0 = options.alpha0; % Parameter of FPGM, can be tuned. 
                             % If options.alpha0 = 0 --> no acceleration, PGM
    alpha(1) = alpha0;
    if options.proj == 1
        H = SimplexProj( H ); % Project columns of H onto the simplex and origin
    elseif options.proj == 0
        H = max(H,0);
    elseif options.proj == 2
        H = SimplexColProj(H'); % Project rows of H onto the simplex
        H = H'; 
    end
    Y = H; % second sequence
    i = 1; 
    % Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F
    eps0 = 0; eps = 1;  
    while i <= options.inner_max_epoch && eps >= options.delta*eps0
        % Previous iterate
        Hp = H; 
        % FGM Coefficients; see Nesterov's book
        alpha(i+1) = ( sqrt(alpha(i)^4 + 4*alpha(i)^2 ) - alpha(i)^2) / (2); 
        beta(i) = alpha(i)*(1-alpha(i))/(alpha(i)^2+alpha(i+1));
        % Projection step
        H = Y - (WtW*Y-WtX) / L;
        if options.proj == 1
            H = SimplexProj( H ); % Project columns of H onto the set {x|x>=0, sum(x) <= 1} 
        elseif options.proj == 0
            H = max(H,0);
        elseif options.proj == 2
            H = SimplexColProj(H'); % Project rows of H onto the simplex
            H = H';
        end
        % `Optimal' linear combination of iterates
        Y = H + beta(i)*(H-Hp); 
        if i == 1
            eps0 = norm(H-Hp,'fro'); 
        end
        eps = norm(H-Hp,'fro'); 
        i = i + 1; 
    end 
end
