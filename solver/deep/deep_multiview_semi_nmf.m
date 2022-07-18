function [x, infos] = deep_multiview_semi_nmf(XX, rank_layers, in_options)
% Deep multi-view semi-NMF.
%
% The problem of interest is defined as
%
%           min sum_{v=1} (alpha^v)^gamma * 
%                   [ || X^v - Z^v_1 * Z^v_2 * ... * Z^v_n * H_n ||_F^2 + beta * tr(H_n * L^v * H_n')],
%           where 
%           {H_n} >= 0, sum_{v=1} alpha^v = 1, and alpha^v >= 0.
%
% Given multi-view matrices XX, factor matrices {Z^v_1, Z^v_2, ..., Z^v_n, H^v_1, ..., H^v_n-1, H_n} are calculated.
%
%
% Inputs:
%       XX          : multi-view matrices to factorize (cell structure)
%       rank_layers : ranks in each layer
%       in_options 
%
%
% Output:
%       x           : non-negative matrix solution, i.e., x.Z (cell), x.H (cell)
%       infos       : log information
%           epoch   : iteration nuber
%           cost    : objective function value
%           optgap  : optimality gap
%           time    : elapsed time
%           grad_calc_count : number of sampled data elements (gradient calculations)
%
% References
%           H. Zhao, Z. Ding, and Y. Fu,
%           "Multi-view clustering via deep matrix factorization",
%           AAAI2017, 2017.
%   
%
% This file is part of NMFLibrary
%
% Originally created by H.Zhao (from the deep semi-nmf code written by G.Trigeorgis).
%
% Change log: 
%
%       Jul. 26, 2018 (Hiroyuki Kasai): Modified code structures.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    n = size(XX, 2);
    
    % set the number of views and rank_layers
    num_of_views = numel(XX);
    num_of_layers = numel(rank_layers);
    
    m_total = 0;
    for i = 1 : num_of_views
        m(i) = size(XX{i}, 1);
        m_total = m_total + m(i);
    end

    % set local options
    local_options.bUpdateZ      = true;
    local_options.bUpdateH      = true;
    local_options.bUpdateLastH  = true;
    local_options.graph_k       = 5;
    local_options.gamma         = 0.1;
    local_options.beta          = 0.01;    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options);

    % initialize
    method_name = 'Deep-Multiview-Semi';       
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
    % initialize for this algorithm
    alpha = ones(num_of_views, 1).*(1/num_of_views); 
    
    Z = cell(num_of_views, num_of_layers);
    H = cell(num_of_views, num_of_layers);
    H_err = cell(num_of_views, num_of_layers);

    A_graph = cell(1, num_of_views);
    D_graph = cell(1, num_of_views);
    L_graph = cell(1, num_of_views);
    w_options = [];
    w_options.k = options.graph_k;
    w_options.WeightMode = 'HeatKernel'; 
    w_options.Metric = 'Euclidean';
    
    
    % initialize Z and H
    for v_ind = 1 : num_of_views
        
        X = XX{v_ind};
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

        A_graph{v_ind} = constructW(X', w_options);
        D_graph{v_ind} = diag(sum(constructW(X', w_options),2));
        L_graph{v_ind} = D_graph{v_ind} - A_graph{v_ind};
    
        if ~isfield(options, 'x_init')
            for i_layer = 1:num_of_layers

                if options.verbose > 1
                    fprintf('### Initializing by %s for view: %d, layer %d ... ', method_name, v_ind, i_layer);
                end

                if i_layer == 1
                    % For the first layer we go linear from X to Z*H, so we use id
                    V = X;
                else 
                    V = H{v_ind, i_layer-1};
                end

                % For the later rank_layers we use nonlinearities as we go from
                % g(H_{k-1}) to Z*H_k     
%                 if 0
%                     [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
%                         seminmf(V, ...
%                         rank_layers(i_layer), ...
%                         'maxiter', options.max_epoch, ...
%                         'bUpdateH', true, 'bUpdateZ', true, 'verbose', options.verbose, 'save', 0, 'fast', 0);                    
%                 else
                    semi_nmf_options.max_iter  = options.max_epoch;
                    semi_nmf_options.bUpdateH  = options.bUpdateH;
                    semi_nmf_options.bUpdateZ  = options.bUpdateZ;
                    semi_nmf_options.verbose   = 0;

                    [semi_nmf_x, ~] = semi_mu_nmf(V, rank_layers(i_layer), semi_nmf_options);
                    Z{v_ind, i_layer} = semi_nmf_x.W;
                    H{v_ind, i_layer} = semi_nmf_x.H;
                %end
                
                if options.verbose > 2 % for debug
                    fprintf('### V: %5.5f, Zi: %5.5f, Hi: %5.5f\n', norm(V), norm(Z{v_ind, i_layer}), norm(H{v_ind, i_layer}));              
                end
                
                if options.verbose > 1
                    fprintf('done\n');
                end            
            end

        else
            Z = options.Z;
            H = options.H;
        end
        
        dnorm0(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^options.gamma, L_graph{v_ind}, options.beta);
        dnorm(v_ind) = dnorm0(v_ind) + 1;
    end  
    

    % store initial info
    clear infos;
    [infos, ~, ~] = store_nmf_info([], [], [], [], options, [], epoch, grad_calc_count, 0);
    f_val = sum(dnorm);
    optgap = f_val - options.f_opt;        
    infos.cost =  f_val;
    infos.optgap = optgap;    

    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end  

    % set start time
    start_time = tic();
    
    % main loop
    while true
        
        % check stop condition
        [stop_flag, reason, max_reached_flag] = check_stop_condition(epoch, infos, options);
        if stop_flag
            display_stop_reason(epoch, infos, options, method_name, reason, max_reached_flag);
            break;
        end
        
        Hm_a = 0; 
        Hm_b = 0;
        for v_ind = 1 : num_of_views
            
            X = XX{v_ind};

            X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

            H_err{v_ind, num_of_layers} = H{v_ind, num_of_layers};
            
            for i_layer = num_of_layers-1:-1:1
                H_err{v_ind, i_layer} = Z{v_ind, i_layer+1} * H_err{v_ind, i_layer+1};
            end        
        
        
            for i = 1 : num_of_layers

                if options.bUpdateZ
                    if i == 1
                        Z{v_ind, i} = X  * pinv(H_err{v_ind, 1});
                    else
                        Z{v_ind, i} = pinv(D') * X * pinv(H_err{v_ind, i});
                    end
                end

                if i == 1
                    D = Z{v_ind,1}';
                else
                    D = Z{v_ind,i}' * D;
                end                

                if options.bUpdateH && (i < num_of_layers || (i == num_of_layers && options.bUpdateLastH))
                    % original one
                    A = D * X;

                    Ap = (abs(A)+A)./2;
                    An = (abs(A)-A)./2;

                    % Hm*A  -> HmA
                    HmA =  options.beta*H{v_ind,i};
                    HmAp = (abs(HmA)+HmA)./2;
                    HmAn = (abs(HmA)-HmA)./2;

                    % original noe
                    B = D * D';
                    Bp = (abs(B)+B)./2;
                    Bn = (abs(B)-B)./2;


                    % Hm*D -> HmD
                    HmD = options.beta*H{v_ind, i};
                    HmDp = (abs(HmD)+HmD)./2;
                    HmDn = (abs(HmD)-HmD)./2;

                    % update graph part
                    H{v_ind, i} = H{v_ind, i} .* sqrt((Ap + Bn* H{v_ind, i} ) ./ max(An + Bp* H{v_ind, i}, 1e-10));

                    % set H{v_ind,n_of_layer} = Hm
                    % update the last consensus layer
                    if i == num_of_layers
                        Hm_a = (alpha(v_ind)^options.gamma)*(Ap + Bn* H{v_ind, i} + HmAp* A_graph{v_ind} + HmDn* D_graph{v_ind}) + Hm_a;
                        Hm_b = (alpha(v_ind)^options.gamma)*(max(An + Bp* H{v_ind, i} + HmAn* A_graph{v_ind} + HmDp* D_graph{v_ind}, 1e-10)) + Hm_b;
                    end
                end
            end

            assert(i == num_of_layers);       
            
        end
        
        % update Hm
        for v_ind = 1 : num_of_views
            H{v_ind, num_of_layers} = H{v_ind, num_of_layers} .* sqrt(Hm_a ./ Hm_b);
        end
        
        if options.verbose > 2 % for debug
            for i_layer = 1 : num_of_layers
                for v_ind = 1 : num_of_layers
                    fprintf('### V: %5.5f, Zi: %5.5f, Hi: %5.5f\n', norm(V), norm(Z{v_ind, i_layer}), norm(H{v_ind, i_layer}));
                end
            end
        end
        
        [dnorm, dnorm_w] = calculate_dnorm(XX, Z, H, L_graph, alpha, options);
        
        %dnorm        
        
        % update alpha
        for v_ind = 1 : num_of_views
            alpha(v_ind) = dnorm_w(v_ind)/sum(dnorm_w);
        end
            

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m_total*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, ~, ~] = store_nmf_info([], [], [], [], options, infos, epoch, grad_calc_count, elapsed_time);  
        f_val = sum(dnorm);
        optgap = f_val - options.f_opt;        
        infos.cost = [infos.cost f_val];
        infos.optgap = [infos.optgap optgap];
        
        % display info
        display_info(method_name, epoch, infos, options);        

    end
    
    x.Z = Z;
    x.H = H;
    
end


function [dnorm, dnorm_w] =  calculate_dnorm(XX, Z, H, L_graph, alpha, options)

    num_of_views = numel(XX);

    for v_ind = 1 : num_of_views

        X = XX{v_ind};
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

        % get the error for each view
        dnorm(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^options.gamma, L_graph{v_ind}, options.beta);

        % the following two lines are used for calculating weight
        tmpNorm = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), 1, L_graph{v_ind}, options.beta);
        dnorm_w(v_ind) = (options.gamma*(tmpNorm))^(1/(1-options.gamma));
    end
    
end


function error = cost_function_graph(X, Z, H, weight, A, beta)

    out = H{numel(H)};
    num_of_layers = numel(Z);
    Z_rec = reconst_Z(Z, num_of_layers);
    f_val = norm(X - Z_rec*H{num_of_layers}, 'fro');
    error = weight * (f_val + beta* trace(out*A*out'));
    
end


% calculate Z = Z_1 * Z_2 * ... * Z_n
function Z_rec = reconst_Z(Z, num_of_layers)

    Z_rec = Z{num_of_layers};
    
    for k = num_of_layers-1 : -1 : 1
        Z_rec =  Z{k} * Z_rec;
    end

end
