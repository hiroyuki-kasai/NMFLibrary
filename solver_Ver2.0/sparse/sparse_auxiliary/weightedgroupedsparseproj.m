% -- Weighted grouped sparse projection -- 
% 
% Definition. Given a vector x and a nonnegative and nonzero vector w, the
% weighted sparsity of x with respect to w is 
%              
%             ||w||_2 - ||x||_w/||x||_2
%   sp_w(x) = --------------------------   in [0,1]. 
%                ||w||_2 - min_j w(j) 
% 
% where ||x||_w = |x|^T w = sum_j |x(j)| w(j). 
% 
% For w=e, this is the sparsity of Hoyer (JMLR, 2001). 
% 
% 
% ******
% input  
% ******
% x{i}, i=1,..,r : as set of r vectors
% s              : target average weighted sparsity for the projected 
%                  vectors xp{i} 
% options: 
%         w         : weights w{i} provides the weights for x{i} to compute
%                     its weighted sparsity (see above) 
%         precision : precision s.t. |sp(x) - s| < precision
%                       -default: 1e-4 
%         linrat    : parameter that guarantees the algorithm to converge
%                     linearly at rate parrl (but could slow down Newton)
%                       -default: 0.9
% ******
% output 
% ******
% Solution xp to 
%  min sum_i ||xp{i}-x{i}||_2 
% such that 
% 1/r * sum_i weighted_sparsity(xp{i}) >= s (average sparsity at least s)
%
% 
% See the paper `Grouped sparse projection', N. Gillis and V. Potluru

function [xp,gxpmu,numiter,newmu] = weightedgroupedsparseproj(x,s,options) 

if nargin <= 2
    options = [];
end
if ~isfield(options,'w')
    for i = 1 : length(x)
        options.w{i} = ones(length(x{i}),1); 
    end
end
if ~isfield(options,'precision')
    options.precision = 1e-3; 
end
if ~isfield(options,'linrat')
    options.linrat = 0.9; 
end
if s < 0 || s > 1
    error('the sparsity parameter has to be in [0,1].'); 
end
% Replace x with |x|, keep the sign sx of its entries  
% and compute the parameter k 
k = 0; 
muup0 = 0; 
r = length(x); 
critmu = []; % set of points where g(mu) is discontinuous
for i = 1 : r 
    sx{i} = sign(x{i}); 
    x{i} = sx{i}.*x{i};
    nwi = norm(options.w{i}); 
    betaim1 = nwi - min( options.w{i} ); 
    k = k + nwi/betaim1; 
    % check critical values of mu where g(mu) is discontinuous, that is, 
    % where the two (or more) largest entries of x{i} are equal to one
    % another 
    [critxi,maxxi] = wcheckcrit(x{i},options.w{i}); 
    muup0 = max(muup0, maxxi*betaim1); 
    critmu = [critmu critxi*betaim1]; 
end 
k = k-r*s; 
[vgmu,xnew,gradg] = wgmu(x,options.w,0); 
if vgmu < k
    xp = x;
    gxpmu = vgmu; 
    numiter = 0; 
    newmu = 0; 
    return;
else
    numiter = 0; 
    mulow = 0;   
    glow = vgmu;   
    muup = muup0; 
    % Initialization on mu using 0, it seems to work best because the
    % slope at zero is rather steep while it is gets falt for large mu
    newmu = 0; 
    gnew = glow; 
    gpnew = gradg; % g'(0) 
    Delta = muup-mulow; 
    while abs(gnew - k) > options.precision*r ... 
                &&  numiter < 100 
        oldmu = newmu;
        % Secant method: 
        % newmu = mulow + (k-glow)*(muup-mulow)/(gup-glow);
        % Bisection: 
        % newmu = (muup+mulow)/2;
        % Newton: 
        newmu = oldmu + (k - gnew) / (gpnew); 
        if newmu >= muup || newmu <= mulow 
            % If Newton goes out of the interval, use bisection
            newmu = (mulow+muup)/2; 
        end
        [gnew,xnew,gpnew] = wgmu(x,options.w,newmu);
        if gnew < k
            gup = gnew;
            xup = xnew;
            muup = newmu;
        else
            glow = gnew;
            mulow = xnew;
            mulow = newmu;
        end
        % Garantees linear convergence 
        if muup - mulow > options.linrat*Delta ... 
                && abs(oldmu-newmu) < (1-options.linrat)*Delta
            newmu = (mulow+muup)/2;
            [gnew,xnew,gpnew] = wgmu(x,options.w,newmu);
            if gnew < k
                gup = gnew;
                xup = xnew;
                muup = newmu;
            else
                glow = gnew;
                mulow = xnew;
                mulow = newmu;
            end
            numiter = numiter+1;
        end
        numiter = numiter+1;
        % Detect discontinuity 
        if ~isempty(critmu) ... 
                && abs(mulow-muup) < abs(newmu)*options.precision ... 
                && min( abs(newmu - critmu) ) < options.precision*newmu
            warning('The objective function is discontinuous around mu^*.')
            xp = xnew;
            gxpmu = gnew;
            break; 
        end
    end
    xp = xnew;
    gxpmu = gnew;
end
% Make sign of xp same as original x + scaling 
for i = 1 : r 
    alpha{i} = xp{i}'*x{i}; 
    xp{i} = alpha{i}*(sx{i}.*xp{i});
end