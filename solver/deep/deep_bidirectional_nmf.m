function [x, infos] = deep_bidirectional_nmf(X, rank_layers, in_options)
% Deep Bidir-Semi-NMF.
%
% The problem of interest is defined as
%
%           min || X - Z_1 * Z_2 * ... * Z_n * H_n ||_F^2,
%           where 
%           {H_n} >= 0.
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
%           G. Trigeorgis, K. Bousmalis, S. Zafeiriou and B. Schuller,
%           "A deep semi-NMF model for learning hidden representations",
%           ICML2014, 2014.
%
%           G. Trigeorgis, K. Bousmalis, S. Zafeiriou and B. Schuller
%           "A deep matrix factorization method for learning attribute representations,"
%           IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), vol.39, no.3, pp.417-429, 2017
%   
%
% This file is part of NMFLibrary
%
% Originally created by G.Trigeorgis.
%
% Change log: 
%
%       Jul. 26, 2018 (Hiroyuki Kasai): Modified code structures.
%
%       Jul. 12, 2022 (Hiroyuki Kasai): Modified code structures.
%


    % set dimensions and samples
    [m, n] = size(X);

    % set the number of rank_layers
    num_of_layers = numel(rank_layers);

    % set local options
    local_options.bUpdateZ  = true;
    local_options.bUpdateH  = true;
    local_options.bUpdateLastH = true;
    local_options.deepZ     = true;
    local_options.deepH     = true;
    local_options.updateH_alg = 'mu';  % 'mu' or 'apg'
    local_options.eval_clustering_acc = 0;
    local_options.classnum      = 0;    
    
    % check input options
    if ~exist('in_options', 'var') || isempty(in_options)
        in_options = struct();
    end     
    % merge options
    options = mergeOptions(get_nmf_default_options(), local_options);   
    options = mergeOptions(options, in_options); 

    % initialize
    method_name = 'Deep-Bidir-SemiNMF';       
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
    H = cell(1, num_of_layers); 
    P = cell(1, num_of_layers);
    Q = cell(1, num_of_layers);     
    %A = cell(1, num_of_layers+1); 
    %B = cell(1, num_of_layers);
    if ~isfield(options, 'x_init')
        
        if options.deepH        
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
                semi_nmf_options.max_iter  = options.max_epoch;
                semi_nmf_options.bUpdateH  = options.bUpdateH;
                semi_nmf_options.bUpdateZ  = options.bUpdateZ;
                semi_nmf_options.verbose   = 0;

                [semi_nmf_x, ~] = semi_mu_nmf(V, rank_layers(i_layer), semi_nmf_options);
                Z{i_layer} = semi_nmf_x.W;
                H{i_layer} = semi_nmf_x.H;

                %fprintf('V: %5.5f, Zi: %5.5f, Hi: %5.5f\n', norm(V), norm(Z{i_layer}), norm(H{i_layer}));              
                if options.verbose > 1
                    fprintf('done\n');
                end            
            end
        end
        
        
        if options.deepZ 
            XT = X';
            for i_layer = 1:num_of_layers

                if options.verbose > 1
                    fprintf('### Initializing by %s for layer %d ... ', method_name, i_layer);
                end

                if i_layer == 1
                    % For the first layer we go linear from X to Z*H, so we use id
                    V = XT;
                else 
                    V = Q{i_layer-1};
                end

                % For the later rank_layers we use nonlinearities as we go from
                % g(H_{k-1}) to Z*H_k              
                semi_nmf_options.max_iter  = options.max_epoch;
                semi_nmf_options.bUpdateH  = options.bUpdateH;
                semi_nmf_options.bUpdateZ  = options.bUpdateZ;
                semi_nmf_options.verbose   = 0;

                [semi_nmf_x, ~] = semi_mu_nmf(V, rank_layers(i_layer), semi_nmf_options);
                P{i_layer} = semi_nmf_x.W;
                Q{i_layer} = semi_nmf_x.H;

                %fprintf('V: %5.5f, Zi: %5.5f, Hi: %5.5f\n', norm(V), norm(Z{i_layer}), norm(H{i_layer}));              
                if options.verbose > 1
                    fprintf('done\n');
                end            
            end
        end        

    else
        Z = options.Z;
        H = options.H;
    end
    
    Hm = H{num_of_layers};
    Qm = Q{num_of_layers};
    
    
    % select disp_freq 
    disp_freq = set_disp_frequency(options);      
    
   
    % store initial info
    clear infos;
    Z_rec = reconst_Z(Z, num_of_layers);  
    
  
   % Concatinated_H = B{1};
    %Concatinated_Q = Z{1}';
    
    
    if options.deepH
        B{num_of_layers} = Hm;
        for i_layer = num_of_layers-1:-1:1
            B{i_layer} = Z{i_layer+1} * B{i_layer+1};
        end
        H_rec = B{1};
        Z_rec = Z{1}; 
        Concatinated_Q = Z{1}';
    elseif options.deepZ
        B{num_of_layers} = Qm;
        for i_layer = num_of_layers-1:-1:1
            B{i_layer} = P{i_layer+1} * B{i_layer+1};
        end
        H_rec = P{1}';
        Z_rec = B{1}';
    elseif options.deepH && options.deepZ
        
    end

    
    %[infos, f_val, optgap] = store_nmf_info(X, Z_rec, Hm, [], options, [], epoch, grad_calc_count, 0);
    [infos, f_val, optgap] = store_nmf_info(X, Z_rec, H_rec, [], options, [], epoch, grad_calc_count, 0);
    
    % evaluate clustering accuracy
    if ~isempty(options.gnd) && options.classnum > 1
        [infos] = store_clustering_accuracy(Hm, options.gnd, options.classnum, infos, options.eval_clustering_num, 0);
    end     

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
        
        % update Z and deep H
        if options.deepH
            Z{1} = Concatinated_Q';
            [Z, Hm, Concatinated_H] = calc_deep_matrices(X, Z, Hm, num_of_layers, options, 1);
        else
            Concatinated_H = P{1}';
        end
        
        
%         [infos, f_val, optgap] = store_nmf_info(X, Z{1}, Concatinated_H, [], options, infos, epoch, grad_calc_count, 0);          
%         % display infos
%         if options.verbose > 1
%             if ~mod(epoch, disp_freq)
%                 fprintf('Deep-SemiNMF (B): Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
%             end
%         end          
%         
        if options.deepZ
            % update H and deep Z
            P{1} = Concatinated_H';
            [P, Qm, Concatinated_Q] = calc_deep_matrices(XT, P, Qm, num_of_layers, options, 0);
        else
            Concatinated_Q = Z{1}';
        end
        
        %fprintf('%e\n', norm(XT-P{1}*Concatinated_Q, 'fro')^2 / 2 );
        
   
       
        

        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % measure gradient calc count
        grad_calc_count = grad_calc_count + m*n;

        % update epoch
        epoch = epoch + 1;         
        
        % store info
        %Z_rec = reconst_Z(Z, num_of_layers);
        %[infos, f_val, optgap] = store_nmf_info(X, Z_rec, Hm, [], options, infos, epoch, grad_calc_count, elapsed_time);  
        [infos, f_val, optgap] = store_nmf_info(X, Concatinated_Q', P{1}', [], options, infos, epoch, grad_calc_count, elapsed_time);          
        
        
%         % display infos
%         if options.verbose > 1
%             if ~mod(epoch, disp_freq)
%                 fprintf('Deep-SemiNMF: Epoch = %04d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
%             end
%         end        

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
    
end


function [Z, Hm, Concatinated_H] = calc_deep_matrices(X, Z, Hm, num_of_layers, options, Hm_nonnegative)

    A = cell(1, num_of_layers+1); 
    B = cell(1, num_of_layers);
    
    m = size(Z, 1);
    
    % B{1} = Z{2} * Z{3} * ... * Z{num_of_layers} * Hm
    % B{2} = Z{3} * ... * Z{num_of_layers} * Hm
    % B{3} = ....
    % ....
    % B{num_of_layers} = Hm
    B{num_of_layers} = Hm;
    for i_layer = num_of_layers-1:-1:1
        B{i_layer} = Z{i_layer+1} * B{i_layer+1};
    end

    %% update Z
    % where Z = Z{1} = X * (B{1})^{-1} due to Z{1} B{1} = X.
    Z{1} = X  * pinv(B{1});    

    A{1} = eye(m);
    % update Z{2}, ... , Z{num_of_layers} 
    for i = 2 : num_of_layers

        % A{1} = 
        % A{2} = Z{1}
        % A{3} = Z{1} * Z{2}
        % A{4} = Z{1} * Z{2} * Z{3}
        % ....
        % A{num_of_layers} = Z{1} * Z{2} * ... * Z{num_of_layers-1}
        if i == 2
            A{i} = Z{i-1};
        else
            A{i} = A{i-1} * Z{i-1};
        end

        Z{i} = pinv(A{i}) * X * pinv(B{i});

    end

    % update Hm
    A{num_of_layers+1} = A{num_of_layers} * Z{num_of_layers};
    if Hm_nonnegative
        if strcmp(options.updateH_alg, 'mu')
            P = A{num_of_layers+1}' * X;
            Pp = (abs(P)+P)./2;
            Pn = (abs(P)-P)./2;

            Q = A{num_of_layers+1}' * A{num_of_layers+1};

            Qp = (abs(Q)+Q)./2;
            Qn = (abs(Q)-Q)./2;

            Hm = Hm .* sqrt((Pp + Qn * Hm) ./ max(Pn + Qp * Hm, 1e-16));        
        else
            % min_H 1/2 | X - A{num_of_layers+1}*H |^2_F 
            % --> min_H 1/2 | X' - H' * A{num_of_layers+1}' |^2_F 
            % ----> min_A 1/2 | X - A * B' |^2_F in "nesterov_mnls(X, B, A,..)"
            [tmpH, ~, ~] = nesterov_mnls(X', A{num_of_layers+1}, Hm', 1, options.apg_maxiter, 'basic');
            Hm = tmpH';
        end 
    else
        Hm = pinv(A{num_of_layers+1}) * X;
    end
    
    %
    B{num_of_layers} = Hm;
    for i_layer = num_of_layers-1:-1:1
        B{i_layer} = Z{i_layer+1} * B{i_layer+1};
    end    
    Concatinated_H = B{1};
    
%     % B{1} = Z{2} * Z{3} * ... * Z{num_of_layers} * H{num_of_layers}
%     % B{2} = Z{3} * ... * Z{num_of_layers} * H{num_of_layers}
%     % B{3} = ....
%     % ....
%     % B{num_of_layers} = H{num_of_layers}
%     B{num_of_layers} = H{num_of_layers};
%     for i_layer = num_of_layers-1:-1:1
%         B{i_layer} = Z{i_layer+1} * B{i_layer+1};
%     end
% 
%     %% update Z
%     % where Z = Z{1} = X * (B{1})^{-1} due to Z{1} B{1} = X.
%     Z{1} = X  * pinv(B{1});    
% 
%     A{1} = eye(m);
%     % update Z{2}, ... , Z{num_of_layers} 
%     for i = 2 : num_of_layers
% 
%         % A{1} = 
%         % A{2} = Z{1}
%         % A{3} = Z{1} * Z{2}
%         % A{4} = Z{1} * Z{2} * Z{3}
%         % ....
%         % A{num_of_layers} = Z{1} * Z{2} * ... * Z{num_of_layers-1}
%         if i == 2
%             A{i} = Z{i-1};
%         else
%             A{i} = A{i-1} * Z{i-1};
%         end
% 
%         Z{i} = pinv(A{i}) * X * pinv(B{i});
% 
%     end
% 
%     % update H
%     A{num_of_layers+1} = A{num_of_layers} * Z{num_of_layers};
%     if strcmp(options.updateH_alg, 'mu')
%         P = A{num_of_layers+1}' * X;
%         Pp = (abs(P)+P)./2;
%         Pn = (abs(P)-P)./2;
% 
%         Q = A{num_of_layers+1}' * A{num_of_layers+1};
% 
%         Qp = (abs(Q)+Q)./2;
%         Qn = (abs(Q)-Q)./2;
% 
% 
%         H{num_of_layers} = H{num_of_layers} .* sqrt((Pp + Qn * H{num_of_layers}) ./ max(Pn + Qp * H{num_of_layers}, 1e-10));        
%     else
%         % min_H 1/2 | X - A{num_of_layers+1}*H |^2_F 
%         % --> min_H 1/2 | X' - H' * A{num_of_layers+1}' |^2_F 
%         % ----> min_A 1/2 | X - A * B' |^2_F in "nesterov_mnls(X, B, A,..)"
%         [tmpH, ~, ~] = nesterov_mnls(X', A{num_of_layers+1}, H{num_of_layers}', 1, 100, 'basic');
%         H{num_of_layers} = tmpH';
%     end


end


% calculate Z = Z_1 * Z_2 * ... * Z_n
function Z_rec = reconst_Z(Z, num_of_layers)

    Z_rec = Z{num_of_layers};
    
    for k = num_of_layers-1 : -1 : 1
        Z_rec =  Z{k} * Z_rec;
    end

end
