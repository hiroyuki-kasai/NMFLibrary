function [x, infos] = deep_ns_nmf(X, rank_layers, in_options)
% Deep Semi-NMF.
%
% The problem of interest is defined as
%
%           
%       where 
%       {H_n} >= 0.
%
% Given a matrix X, factor matrices {Z_1, Z_2, ..., Z_n, H_n} are calculated.
%
%
% Inputs:
%       X           : (m x n) matrix to factorize
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
%
%
% This file is part of NMFLibrary
%
% Created by H.Kasai on Jul. 28, 2018
%
% Change log: 
%
%       Aug. 03, 2018 (Hiroyuki Kasai): Modified code structures.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    m = size(X, 1);
    n = size(X, 2);
    
    % set the number of rank_layers
    num_of_layers = numel(rank_layers);

    % set local options
    local_options.theta         = 0.5;
    local_options.bUpdateZ      = true;
    local_options.bUpdateH      = true;
    local_options.bUpdateLastH  = true;
    local_options.norm_w        = 1;
    local_options.update_alg    = 'mu';  % 'mu' or 'apg' 
    local_options.apg_maxiter   = 100;
    local_options.eval_clustering_acc = 0;
    local_options.eval_clustering_num = 10;
    local_options.classnum      = 0;
    local_options.initialize_max_epoch = 100;
    
    
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 
    
    % initialize
    method_name = 'Deep-nsNMF';    
    epoch = 0;    
    grad_calc_count = 0;

    if options.verbose > 0
        fprintf('# %s: started ...\n', method_name);           
    end      
    
    if ~isempty(options.gnd) && options.classnum > 1
        options.eval_clustering_acc = 1;
    end    
    
    % initialize Z and H
    Z = cell(1, num_of_layers);
    S = cell(1, num_of_layers);    
    H = cell(1, num_of_layers); 
    if ~isfield(options, 'x_init')
        
        for i_layer = 1:num_of_layers

            if options.verbose > 1
                fprintf('### Initializing by %s for layer %d ... ', method_name, i_layer);
            end

            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = X;
            else 
                V = H{i_layer-1};
            end

            % For the later rank_layers we use nonlinearities as we go from
            % g(H_{k-1}) to Z*H_k    
            %ns_nmf_options.max_epoch    = min(options.max_epoch, 100);
            ns_nmf_options.max_epoch    = options.initialize_max_epoch;
            ns_nmf_options.bUpdateH     = options.bUpdateH;
            ns_nmf_options.bUpdateZ     = options.bUpdateZ;
            ns_nmf_options.theta        = options.theta;
            ns_nmf_options.update_alg   = 'apg';
            ns_nmf_options.apg_maxiter  = options.apg_maxiter;
            ns_nmf_options.norm_w       = options.norm_w;
            ns_nmf_options.verbose      = 0;

            [ns_nmf_x, ~] = ns_nmf(V, rank_layers(i_layer), ns_nmf_options);
            Z{i_layer} = ns_nmf_x.W;
            S{i_layer} = ns_nmf_x.S;
            H{i_layer} = ns_nmf_x.H;

            %fprintf('V: %5.5f, Zi: %5.5f, Hi: %5.5f\n', norm(V), norm(Z{i_layer}), norm(H{i_layer}));              
            if options.verbose > 1
                fprintf('done\n');
            end            
        end
    end


    Hm = H{num_of_layers};


    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
   
    % store initial info
    clear infos;
    
    % reconstruct B (= B{1} = S{1} * Z{2}*S{2} * Z{3}*S{3} * ... * Z{num_of_layers}*S{num_of_layers} * Hm)
    B{num_of_layers} = S{num_of_layers} * Hm;
    for i_layer = num_of_layers-1:-1:1
        B{i_layer} = S{i_layer} * Z{i_layer+1} * B{i_layer+1};
    end
    H1 = B{1};
    [infos, f_val, optgap] = store_nmf_info(X, Z{1}, H1, [], options, [], epoch, grad_calc_count, 0);

    if options.verbose > 1
        fprintf('%s: Epoch = 0000, cost = %.16e, optgap = %.4e\n', method_name, f_val, optgap); 
    end  
    
    % evaluate clustering accuracy
    if ~isempty(options.gnd) && options.classnum > 1
        [infos] = store_clustering_accuracy(Hm, options.gnd, options.classnum, infos, options.eval_clustering_num, 0);
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
        
        % update Z and deep H
        [Z, H1, Hm] = calc_deep_matrices(X, Z, Hm, S, num_of_layers, options);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        [infos, f_val, optgap] = store_nmf_info(X, Z{1}, H1, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        
        % evaluate clustering accuracy
        if options.eval_clustering_acc
            [infos] = store_clustering_accuracy(Hm, options.gnd, options.classnum, infos, options.eval_clustering_num, epoch);
        end        
        
        % display infos
        if options.verbose > 1
            if ~mod(epoch, disp_freq)

                if ~options.eval_clustering_acc
                    fprintf('%s: Epoch = %04d, cost = %.16e, optgap = %.4e\n', method_name, epoch, f_val, optgap);
                else
                    fprintf('%s: Epoch = %04d, cost = %.16e, optgap = %.4e, acc = %.4f, nmi = %.4f, purity = %.4f, f = %.4f\n', ...
                        method_name, epoch, f_val, optgap, infos.clustering_acc(end).acc, infos.clustering_acc(end).nmi, infos.clustering_acc(end).purity, infos.clustering_acc(end).f_val);
                end

            end 
            
        elseif options.verbose == 1
            textwaitbar(epoch, options.max_epoch, '  progress');
        end        
    end
    
    H{num_of_layers} = Hm;

    x.Z = Z;
    x.H = H;
    x.S = S;
end


function [Z, B1, Hm] = calc_deep_matrices(X, Z, Hm, S, num_of_layers, options)

    A = cell(1, num_of_layers+1); 
    B = cell(1, num_of_layers);
    
    m = size(Z{1}, 1);
    n = size(Z{1}, 2);
    
    % B{1} = S{1} * Z{2}*S{2} * Z{3}*S{3} * ... * Z{num_of_layers}*S{num_of_layers} * Hm
    % B{2} = S{2} * Z{3}*S{3} * ... * Z{num_of_layers}*S{num_of_layers} * Hm
    % B{3} = S{3} * Z{4}*S{4} * ... * Z{num_of_layers}*S{num_of_layers} * Hm
    % ....
    % B{num_of_layers} = Hm
    B{num_of_layers} = S{num_of_layers} * Hm;
    for i_layer = num_of_layers-1:-1:1
        B{i_layer} = S{i_layer} * Z{i_layer+1} * B{i_layer+1};
    end

    % update Z{1]
    if strcmp(options.update_alg, 'apg') 
        % min_H 1/2 | X - Z_1 * B{1} |^2_F 
        % --> min_A 1/2 | X - A * B' |^2_F in "nesterov_mnls(X, B, A,..)"
        [Z{1}, ~, ~] = nesterov_mnls_general(X, [], B{1}', Z{1}, 1, options.apg_maxiter, 'basic');  
        %iter_num
    else
        Z{1} = Z{1} .* (X*B{1}') ./ (Z{1}*(B{1}*B{1}') + 1e-9);
    end
    
    if options.norm_w
        %Z{1} = bsxfun(@rdivide,Z{1},sqrt(sum(Z{1}.^2,1)));
        [Z{1}, ~] = normalize_data(Z{1}, [], 'std');
    end

    % update Z{2}, ... , Z{num_of_layers} 
    A{1} = eye(m);
    for i = 2 : num_of_layers

        % A{1} = 
        % A{2} = Z{1}*S{1}
        % A{3} = Z{1}*S{1} * Z{2}*S{2}
        % A{4} = Z{1}*S{1} * Z{2}*S{2} * Z{3}*S{3}
        % ....
        % A{num_of_layers} = Z{1}*S{1} * Z{2}*S{2} * ... * Z{num_of_layers-1}*S{num_of_layers-1}
        if i == 2
            A{i} = Z{i-1} * S{i-1};
        else
            A{i} = A{i-1} * Z{i-1} * S{i-1};
        end

        %Z{i} = pinv(A{i}) * X * pinv(B{i});
        [Z{i}, ~, ~] = nesterov_mnls_general(X, A{i}, B{i}', Z{i}, 1, options.apg_maxiter, 'basic');
        %iter_num
        
        if options.norm_w        
            %Z{i} = bsxfun(@rdivide,Z{i},sqrt(sum(Z{i}.^2,1)));
            [Z{i}, ~] = normalize_data(Z{i}, [], 'std');
        end

    end

    % update H
    if num_of_layers ~= 1
        A{num_of_layers+1} = A{num_of_layers} * Z{num_of_layers} * S{num_of_layers};
    else
        A{num_of_layers+1} = Z{num_of_layers} * S{num_of_layers};
    end
    if strcmp(options.update_alg, 'apg')
        % min_H 1/2 | X - A{num_of_layers+1}*H |^2_F 
        % ----> min_A 1/2 | X - C * A |^2_F in "nesterov_mnls(X, C, [], A,..)"
        %[tmpH, iter_num, ~] = nesterov_mnls_general(X', [], A{num_of_layers+1}, Hm', 1, options.apg_maxiter, 'basic');
        %Hm = tmpH';
        [Hm, iter_num, ~] = nesterov_mnls_general(X, A{num_of_layers+1}, [], Hm, 1, options.apg_maxiter, 'basic');
        %iter_num
    else
        P = A{num_of_layers+1}' * X;
        Pp = (abs(P)+P)./2;
        Pn = (abs(P)-P)./2;

        Q = A{num_of_layers+1}' * A{num_of_layers+1};

        Qp = (abs(Q)+Q)./2;
        Qn = (abs(Q)-Q)./2;

        Hm = Hm .* sqrt((Pp + Qn * Hm) ./ max(Pn + Qp * Hm, 1e-10));            
    end
    
    
    % reconstruct B (= B{1} = S{1} * Z{2}*S{2} * Z{3}*S{3} * ... * Z{num_of_layers}*S{num_of_layers} * Hm)
    B{num_of_layers} = S{num_of_layers} * Hm;
    for i_layer = num_of_layers-1:-1:1
        B{i_layer} = S{i_layer} * Z{i_layer+1} * B{i_layer+1};
    end
    B1 = B{1};
    
end


% calculate Z = Z_1 * Z_2 * ... * Z_n
function Z_rec = reconst_Z(Z, num_of_layers)

    Z_rec = Z{num_of_layers};
    
    for k = num_of_layers-1 : -1 : 1
        Z_rec =  Z{k} * Z_rec;
    end

end
